
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, degree
from torch_scatter import scatter_mean
import numpy as np
from typing import Dict, List, Optional, Callable, Union
import warnings
warnings.filterwarnings('ignore')


# Import gradient conflict module
try:
    from gradient_conflict_fairness import ConflictAwareWeightScheduler
    GRADIENT_CONFLICT_AVAILABLE = True
except ImportError:
    GRADIENT_CONFLICT_AVAILABLE = False
    warnings.warn("Gradient conflict module not available. Install to use conflict-aware scheduling.")


class ProgressiveFairnessLoss(nn.Module):
    """
    Progressive Fairness Regularization Loss
    ----------------------------------------------------
    - Can be enhanced with gradient conflict analysis (use_gradient_conflict=True)
    """

    def __init__(self,
                 schedule_type: str = 'linear',
                 alpha: float = 1.0,
                 beta: float = 0.1,
                 learnable_schedule: bool = False,
                 fairness_metrics: List[str] = ['representation_mmd', 'conditional_mmd_eod'],
                 class_weights: Optional[torch.Tensor] = None,
                 # MMD parameters
                 mmd_kernel: str = 'rbf',
                 kernel_mul: float = 2.0,
                 kernel_num: int = 5,
                 apply_mmd_on: str = 'embeddings',
                 # Gradient conflict parameters
                 use_gradient_conflict: bool = False,
                 conflict_strategy: str = 'inverse',
                 conflict_threshold: float = -0.1,
                 conflict_smoothing: float = 0.9,
                 use_smoothed_conflicts: bool = True):
        """
        Initialize Progressive Fairness Loss

        Args:
            schedule_type: Type of progressive schedule ('linear', 'exponential', 'sigmoid', 'power', 'inverse_sqrt')
            alpha: Scaling parameter for schedule function
            beta: Offset parameter for schedule function
            learnable_schedule: Whether to learn schedule parameters
            fairness_metrics: List of fairness metrics
                             Options:
                             - 'representation_mmd': Unconditional MMD (correlates with SPD)
                             - 'conditional_mmd_eod': Conditional MMD on Y=1 (correlates with EOD)
                             - 'conditional_mmd_equalized_odds': Conditional MMD on Y=0,1 (correlates with Equalized Odds)
                             - 'representation_variance': Variance fairness
                             - 'representation_covariance': Covariance fairness
            class_weights: Weights for handling class imbalance
            mmd_kernel: Kernel type for MMD ('rbf' or 'linear')
            kernel_mul: Kernel bandwidth multiplier for multi-scale RBF
            kernel_num: Number of kernels for multi-kernel MMD
            apply_mmd_on: Where to apply MMD ('embeddings' or 'logits' - logits recommended for stronger correlation)
            use_gradient_conflict: If True, adapt weights based on gradient conflicts (recommended for GC-PFR)
            conflict_strategy: How to convert conflicts to weights ('inverse', 'threshold', 'exponential')
            conflict_threshold: Threshold below which conflicts are severe (default: -0.1)
            conflict_smoothing: Smoothing factor for conflict history 0-1 (default: 0.9)
            use_smoothed_conflicts: Whether to use smoothed conflict estimates
        """
        super().__init__()

        self.schedule_type = schedule_type
        self.alpha = alpha
        self.beta = beta
        self.fairness_metrics = fairness_metrics
        self.class_weights = class_weights
        self.use_gradient_conflict = use_gradient_conflict

        # Store MMD parameters
        self.mmd_kernel = mmd_kernel
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.apply_mmd_on = apply_mmd_on

        # Learnable schedule parameters
        if learnable_schedule:
            self.alpha_param = nn.Parameter(torch.tensor(alpha))
            self.beta_param = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('alpha_param', torch.tensor(alpha))
            self.register_buffer('beta_param', torch.tensor(beta))

        # Gradient conflict scheduler
        if use_gradient_conflict:
            if not GRADIENT_CONFLICT_AVAILABLE:
                raise ImportError("Gradient conflict module not available. Cannot use gradient conflict scheduling.")

            self.conflict_scheduler = ConflictAwareWeightScheduler(
                base_schedule_type=schedule_type,
                alpha=alpha,
                beta=beta,
                conflict_strategy=conflict_strategy,
                use_smoothed_conflicts=use_smoothed_conflicts
            )
            self.conflict_scheduler.conflict_analyzer.conflict_threshold = conflict_threshold
            self.conflict_scheduler.conflict_analyzer.smoothing_factor = conflict_smoothing
        else:
            self.conflict_scheduler = None

        # Store metrics for analysis
        self.layer_weights_history = []
        self.fairness_violations_history = []
        self.gradient_conflicts_history = []

        # Correlation tracking - stores MMD and fairness metric values across epochs
        self.mmd_values_history = []  # Unconditional MMD
        self.conditional_mmd_values_history = []  # Conditional MMD (EOD)
        self.spd_values_history = []  # Statistical Parity Difference
        self.eod_values_history = []  # Equal Opportunity Difference
        self.equalized_odds_values_history = []  # Equalized Odds violation

    def get_layer_weight(self, layer_idx: int, total_layers: int) -> torch.Tensor:
        """
        Compute progressive weight for a given layer

        Args:
            layer_idx: Current layer index (0-based)
            total_layers: Total number of layers

        Returns:
            Weight for this layer (higher for early layers)
        """
        # Convert to 1-based indexing for mathematical clarity
        l = layer_idx + 1

        if self.schedule_type == 'linear':
            # Linear decay: w(l) = α/l + β
            weight = self.alpha_param / l + self.beta_param

        elif self.schedule_type == 'exponential':
            # Exponential decay: w(l) = α * exp(-β * l)
            weight = self.alpha_param * torch.exp(-self.beta_param * l)

        elif self.schedule_type == 'sigmoid':
            # Sigmoid schedule: w(l) = σ(α/l - β)
            weight = torch.sigmoid(self.alpha_param / l - self.beta_param)

        elif self.schedule_type == 'power':
            # Power law decay: w(l) = α * l^(-β)
            weight = self.alpha_param * (l ** (-self.beta_param))

        elif self.schedule_type == 'inverse_sqrt':
            # Inverse square root: w(l) = α / sqrt(l + β)
            weight = self.alpha_param / torch.sqrt(l + self.beta_param)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Ensure non-negative weights
        weight = torch.clamp(weight, min=1e-6)

        return weight

    def compute_representation_mmd(self,
                                  embeddings: torch.Tensor,
                                  sensitive_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute Unconditional Maximum Mean Discrepancy (MMD) between group embeddings

        This is the UNCONDITIONAL MMD that measures if learned representations are fair.

        Correlation: This metric correlates with Statistical Parity Difference (SPD)
        SPD = |P(Ŷ=1|S=0) - P(Ŷ=1|S=1)|

        NOTE: This now uses proper kernel-based MMD instead of simple mean difference.

        Args:
            embeddings: Node embeddings/representations
            sensitive_attr: Sensitive attributes

        Returns:
            MMD value
        """
        # Use the proper MMD computation
        return self.compute_proper_mmd(
            embeddings,
            sensitive_attr,
            kernel_type=self.mmd_kernel,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num
        )

    def compute_representation_variance_fairness(self,
                                                embeddings: torch.Tensor,
                                                sensitive_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute variance fairness: Both groups should have similar within-group variance

        This ensures not just that means are similar, but that the spread/diversity
        of representations is similar across groups.
        """
        mask_s0 = (sensitive_attr == 0)
        mask_s1 = (sensitive_attr == 1)

        if not mask_s0.any() or not mask_s1.any():
            return torch.tensor(0.0, device=embeddings.device)

        emb_s0 = embeddings[mask_s0]
        emb_s1 = embeddings[mask_s1]

        # Variance for each group
        var_s0 = emb_s0.var(dim=0).mean() if emb_s0.size(0) > 1 else torch.tensor(0.0, device=embeddings.device)
        var_s1 = emb_s1.var(dim=0).mean() if emb_s1.size(0) > 1 else torch.tensor(0.0, device=embeddings.device)

        # Fairness violation: difference in variances
        var_diff = torch.abs(var_s0 - var_s1)

        return var_diff

    def compute_representation_covariance_fairness(self,
                                                  embeddings: torch.Tensor,
                                                  sensitive_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance fairness: Group covariance matrices should be similar

        This captures higher-order statistical properties of the representations.
        """
        mask_s0 = (sensitive_attr == 0)
        mask_s1 = (sensitive_attr == 1)

        if not mask_s0.any() or not mask_s1.any():
            return torch.tensor(0.0, device=embeddings.device)

        emb_s0 = embeddings[mask_s0]
        emb_s1 = embeddings[mask_s1]

        if emb_s0.size(0) < 2 or emb_s1.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Center embeddings
        emb_s0_centered = emb_s0 - emb_s0.mean(dim=0)
        emb_s1_centered = emb_s1 - emb_s1.mean(dim=0)

        # Covariance matrices
        cov_s0 = torch.mm(emb_s0_centered.T, emb_s0_centered) / emb_s0.size(0)
        cov_s1 = torch.mm(emb_s1_centered.T, emb_s1_centered) / emb_s1.size(0)

        # Frobenius norm of difference
        cov_diff = torch.norm(cov_s0 - cov_s1, p='fro')

        return cov_diff

    def compute_proper_mmd(self,
                          embeddings: torch.Tensor,
                          sensitive_attr: torch.Tensor,
                          kernel_type: str = 'rbf',
                          kernel_mul: float = 2.0,
                          kernel_num: int = 5,
                          max_samples: int = 1000) -> torch.Tensor:
        """
        Compute PROPER Maximum Mean Discrepancy using kernel trick

        

        Args:
            embeddings: Node embeddings/representations or logits
            sensitive_attr: Sensitive attributes (0 or 1)
            kernel_type: 'rbf' (Gaussian) or 'linear'
            kernel_mul: Kernel bandwidth multiplier for multi-scale
            kernel_num: Number of kernels for multi-kernel MMD
            max_samples: Maximum samples per group to avoid OOM (default: 1000)

        Returns:
            MMD value (scalar tensor)
        """
        mask_s0 = (sensitive_attr == 0)
        mask_s1 = (sensitive_attr == 1)

        if not mask_s0.any() or not mask_s1.any():
            return torch.tensor(0.0, device=embeddings.device)

        X = embeddings[mask_s0]  # Group 0
        Y = embeddings[mask_s1]  # Group 1

        n = X.size(0)
        m = Y.size(0)

        # Subsample if too many samples (to avoid OOM)
        if n > max_samples:
            indices = torch.randperm(n, device=X.device)[:max_samples]
            X = X[indices]
            n = max_samples

        if m > max_samples:
            indices = torch.randperm(m, device=Y.device)[:max_samples]
            Y = Y[indices]
            m = max_samples

        if kernel_type == 'rbf':
            # Compute pairwise squared distances
            XX = torch.mm(X, X.t())
            YY = torch.mm(Y, Y.t())
            XY = torch.mm(X, Y.t())

            X_sqnorms = torch.diag(XX).unsqueeze(1)
            Y_sqnorms = torch.diag(YY).unsqueeze(1)

            # Squared distances: ||x - x'||²
            K_XX = X_sqnorms + X_sqnorms.t() - 2 * XX
            K_YY = Y_sqnorms + Y_sqnorms.t() - 2 * YY
            K_XY = X_sqnorms + Y_sqnorms.t() - 2 * XY

            # Compute bandwidth using median heuristic
            bandwidth = torch.median(K_XY[K_XY > 0])
            if bandwidth == 0 or torch.isnan(bandwidth):
                bandwidth = 1.0

            # Multi-kernel MMD with multiple bandwidths
            mmd_value = 0.0
            for i in range(kernel_num):
                bandwidth_i = bandwidth * (kernel_mul ** (i - kernel_num // 2))

                # Apply RBF kernel: K(x,x') = exp(-||x-x'||² / (2*σ²))
                K_XX_kernel = torch.exp(-K_XX / (2 * bandwidth_i))
                K_YY_kernel = torch.exp(-K_YY / (2 * bandwidth_i))
                K_XY_kernel = torch.exp(-K_XY / (2 * bandwidth_i))

                # MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
                # Remove diagonal elements to get unbiased estimate
                K_XX_sum = (K_XX_kernel.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
                K_YY_sum = (K_YY_kernel.sum() - m) / (m * (m - 1)) if m > 1 else 0.0
                K_XY_sum = K_XY_kernel.sum() / (n * m)

                mmd_value += K_XX_sum + K_YY_sum - 2 * K_XY_sum

            mmd_value = mmd_value / kernel_num

        elif kernel_type == 'linear':
            # Linear kernel: k(x,y) = x^T y
            K_XX = torch.mm(X, X.t())
            K_YY = torch.mm(Y, Y.t())
            K_XY = torch.mm(X, Y.t())

            # Unbiased estimator
            K_XX_sum = (K_XX.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
            K_YY_sum = (K_YY.sum() - m) / (m * (m - 1)) if m > 1 else 0.0
            K_XY_sum = K_XY.sum() / (n * m)

            mmd_value = K_XX_sum + K_YY_sum - 2 * K_XY_sum

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Ensure non-negative (numerical errors can cause slightly negative values)
        mmd_value = torch.clamp(mmd_value, min=0.0)

        # Return MMD (not MMD²)
        result = torch.sqrt(mmd_value + 1e-8)  # Add epsilon for numerical stability

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def compute_conditional_mmd_eod(self,
                                    embeddings: torch.Tensor,
                                    sensitive_attr: torch.Tensor,
                                    labels: torch.Tensor,
                                    kernel_type: str = 'rbf') -> torch.Tensor:
        """
        Compute Conditional MMD for Equal Opportunity Difference (EOD)

        This computes MMD only among TRULY POSITIVE nodes (Y=1)

        Correlation: This metric correlates with Equal Opportunity Difference (EOD)
        EOD = |P(Ŷ=1|Y=1,S=0) - P(Ŷ=1|Y=1,S=1)|

        Args:
            embeddings: Node embeddings/representations or logits
            sensitive_attr: Sensitive attributes
            labels: Ground truth labels (REQUIRED - condition on Y=1)
            kernel_type: Kernel type for MMD computation

        Returns:
            Conditional MMD value for positive class
        """
        # Filter to truly positive nodes (Y=1)
        positive_mask = (labels == 1)

        if not positive_mask.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Get embeddings and sensitive attributes for positive nodes only
        pos_embeddings = embeddings[positive_mask]
        pos_sensitive = sensitive_attr[positive_mask]

        # Compute MMD among positive nodes
        return self.compute_proper_mmd(pos_embeddings, pos_sensitive, kernel_type)

    def compute_conditional_mmd_equalized_odds(self,
                                              embeddings: torch.Tensor,
                                              sensitive_attr: torch.Tensor,
                                              labels: torch.Tensor,
                                              kernel_type: str = 'rbf') -> torch.Tensor:
        """
        Compute Conditional MMD for Equalized Odds

        This computes MMD separately for Y=0 and Y=1, then sums them

        Correlation: This metric correlates with Equalized Odds
        Equalized Odds requires both:
        - Equal TPR: P(Ŷ=1|Y=1,S=0) = P(Ŷ=1|Y=1,S=1)
        - Equal FPR: P(Ŷ=1|Y=0,S=0) = P(Ŷ=1|Y=0,S=1)

        Args:
            embeddings: Node embeddings/representations or logits
            sensitive_attr: Sensitive attributes
            labels: Ground truth labels (REQUIRED)
            kernel_type: Kernel type for MMD computation

        Returns:
            Sum of conditional MMD for both classes
        """
        # MMD for positive class (Y=1)
        positive_mask = (labels == 1)
        if positive_mask.any():
            pos_embeddings = embeddings[positive_mask]
            pos_sensitive = sensitive_attr[positive_mask]
            mmd_positive = self.compute_proper_mmd(pos_embeddings, pos_sensitive, kernel_type)
        else:
            mmd_positive = torch.tensor(0.0, device=embeddings.device)

        # MMD for negative class (Y=0)
        negative_mask = (labels == 0)
        if negative_mask.any():
            neg_embeddings = embeddings[negative_mask]
            neg_sensitive = sensitive_attr[negative_mask]
            mmd_negative = self.compute_proper_mmd(neg_embeddings, neg_sensitive, kernel_type)
        else:
            mmd_negative = torch.tensor(0.0, device=embeddings.device)

        # Sum both (can also weight them differently if needed)
        return mmd_positive + mmd_negative

    def compute_mmd_on_logits(self,
                             logits: torch.Tensor,
                             sensitive_attr: torch.Tensor,
                             labels: Optional[torch.Tensor] = None,
                             conditional: bool = False) -> torch.Tensor:
        """
        Compute MMD on model logits

        Args:
            logits: Model output BEFORE activation (raw scores)
            sensitive_attr: Sensitive attributes
            labels: Ground truth labels (for conditional MMD)
            conditional: If True, compute conditional MMD (for EOD)

        Returns:
            MMD value
        """
        # Apply log-softmax transformation (as in Stanford study)
        if logits.dim() == 1:
            # Binary classification - logits are scalars
            log_probs = logits.unsqueeze(-1)
        else:
            # Multi-class - apply log-softmax
            log_probs = F.log_softmax(logits, dim=-1)

        if conditional and labels is not None:
            # Conditional MMD for EOD
            return self.compute_conditional_mmd_eod(
                log_probs, sensitive_attr, labels, kernel_type=self.mmd_kernel
            )
        else:
            # Unconditional MMD for SPD
            return self.compute_proper_mmd(
                log_probs, sensitive_attr, kernel_type=self.mmd_kernel,
                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num
            )

    def compute_fairness_metrics(self,
                                predictions: torch.Tensor,
                                labels: torch.Tensor,
                                sensitive_attr: torch.Tensor) -> Dict[str, float]:
        """
        Compute actual fairness metrics: SPD, EOD, Equalized Odds

        Args:
            predictions: Model predictions (probabilities or binary predictions)
            labels: Ground truth labels
            sensitive_attr: Sensitive attributes

        Returns:
            Dictionary containing SPD, EOD, and Equalized Odds violations
        """
        # Ensure predictions are binary
        if predictions.dtype == torch.float:
            binary_preds = (predictions > 0.5).float()
        else:
            binary_preds = predictions.float()

        # Statistical Parity Difference (SPD)
        # SPD = |P(Ŷ=1|S=0) - P(Ŷ=1|S=1)|
        mask_s0 = (sensitive_attr == 0)
        mask_s1 = (sensitive_attr == 1)

        if mask_s0.any() and mask_s1.any():
            pred_rate_s0 = binary_preds[mask_s0].mean().item()
            pred_rate_s1 = binary_preds[mask_s1].mean().item()
            spd = abs(pred_rate_s0 - pred_rate_s1)
        else:
            spd = 0.0

        # Equal Opportunity Difference (EOD)
        # EOD = |P(Ŷ=1|Y=1,S=0) - P(Ŷ=1|Y=1,S=1)|
        positive_mask = (labels == 1)
        if positive_mask.any():
            pos_s0_mask = positive_mask & mask_s0
            pos_s1_mask = positive_mask & mask_s1

            if pos_s0_mask.any() and pos_s1_mask.any():
                tpr_s0 = binary_preds[pos_s0_mask].mean().item()
                tpr_s1 = binary_preds[pos_s1_mask].mean().item()
                eod = abs(tpr_s0 - tpr_s1)
            else:
                eod = 0.0
        else:
            eod = 0.0

        # Equalized Odds
        # Max of TPR difference and FPR difference
        negative_mask = (labels == 0)
        if negative_mask.any():
            neg_s0_mask = negative_mask & mask_s0
            neg_s1_mask = negative_mask & mask_s1

            if neg_s0_mask.any() and neg_s1_mask.any():
                fpr_s0 = binary_preds[neg_s0_mask].mean().item()
                fpr_s1 = binary_preds[neg_s1_mask].mean().item()
                fpr_diff = abs(fpr_s0 - fpr_s1)
            else:
                fpr_diff = 0.0
        else:
            fpr_diff = 0.0

        equalized_odds = max(eod, fpr_diff)

        return {
            'spd': spd,
            'eod': eod,
            'equalized_odds': equalized_odds
        }

    def update_correlation_metrics(self,
                                   embeddings: torch.Tensor,
                                   predictions: torch.Tensor,
                                   labels: torch.Tensor,
                                   sensitive_attr: torch.Tensor):
        """
        Update correlation tracking by computing both MMD values and fairness metrics

        This should be called during evaluation to track correlation over time.

        Args:
            embeddings: Node embeddings from the model
            predictions: Model predictions (probabilities)
            labels: Ground truth labels
            sensitive_attr: Sensitive attributes
        """
        with torch.no_grad():
            # Compute MMD values
            if 'representation_mmd' in self.fairness_metrics:
                mmd_val = self.compute_representation_mmd(embeddings, sensitive_attr).item()
                self.mmd_values_history.append(mmd_val)

            if 'conditional_mmd_eod' in self.fairness_metrics:
                cond_mmd_val = self.compute_conditional_mmd_eod(
                    embeddings, sensitive_attr, labels, kernel_type=self.mmd_kernel
                ).item()
                self.conditional_mmd_values_history.append(cond_mmd_val)

            # Compute actual fairness metrics
            fairness_metrics = self.compute_fairness_metrics(predictions, labels, sensitive_attr)
            self.spd_values_history.append(fairness_metrics['spd'])
            self.eod_values_history.append(fairness_metrics['eod'])
            self.equalized_odds_values_history.append(fairness_metrics['equalized_odds'])

    def get_correlation_statistics(self) -> Dict[str, float]:
        """
        Compute correlation coefficients between MMD and fairness metrics

        Returns:
            Dictionary containing correlation coefficients
        """
        if len(self.mmd_values_history) < 2:
            return {
                'mmd_spd_correlation': 0.0,
                'conditional_mmd_eod_correlation': 0.0,
                'note': 'Need at least 2 epochs of data for correlation'
            }

        correlations = {}

        # Compute correlation between unconditional MMD and SPD
        if len(self.mmd_values_history) > 0 and len(self.spd_values_history) > 0:
            mmd_arr = np.array(self.mmd_values_history)
            spd_arr = np.array(self.spd_values_history)
            if len(mmd_arr) == len(spd_arr) and len(mmd_arr) >= 2:
                corr_matrix = np.corrcoef(mmd_arr, spd_arr)
                correlations['mmd_spd_correlation'] = float(corr_matrix[0, 1])

        # Compute correlation between conditional MMD and EOD
        if len(self.conditional_mmd_values_history) > 0 and len(self.eod_values_history) > 0:
            cond_mmd_arr = np.array(self.conditional_mmd_values_history)
            eod_arr = np.array(self.eod_values_history)
            if len(cond_mmd_arr) == len(eod_arr) and len(cond_mmd_arr) >= 2:
                corr_matrix = np.corrcoef(cond_mmd_arr, eod_arr)
                correlations['conditional_mmd_eod_correlation'] = float(corr_matrix[0, 1])

        # Add recent values for context
        if len(self.mmd_values_history) > 0:
            correlations['latest_mmd'] = self.mmd_values_history[-1]
            correlations['latest_spd'] = self.spd_values_history[-1]

        if len(self.conditional_mmd_values_history) > 0:
            correlations['latest_conditional_mmd'] = self.conditional_mmd_values_history[-1]
            correlations['latest_eod'] = self.eod_values_history[-1]

        return correlations

    def compute_layer_fairness(self,
                              embeddings: torch.Tensor,
                              sensitive_attr: torch.Tensor,
                              labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute representation-based fairness violations for a layer

        Args:
            embeddings: Layer embeddings/representations
            sensitive_attr: Sensitive attributes
            labels: Ground truth labels (REQUIRED for conditional MMD metrics)

        Returns:
            Dictionary of fairness violations
        """

        violations = {}

        # UNCONDITIONAL fairness metrics (no labels needed)
        if 'representation_mmd' in self.fairness_metrics:
            violations['representation_mmd'] = self.compute_representation_mmd(
                embeddings, sensitive_attr
            )

        if 'representation_variance' in self.fairness_metrics:
            violations['representation_variance'] = self.compute_representation_variance_fairness(
                embeddings, sensitive_attr
            )

        if 'representation_covariance' in self.fairness_metrics:
            violations['representation_covariance'] = self.compute_representation_covariance_fairness(
                embeddings, sensitive_attr
            )

        # CONDITIONAL fairness metrics (labels REQUIRED)
        if labels is not None:
            if 'conditional_mmd_eod' in self.fairness_metrics:
                violations['conditional_mmd_eod'] = self.compute_conditional_mmd_eod(
                    embeddings, sensitive_attr, labels, kernel_type=self.mmd_kernel
                )

            if 'conditional_mmd_equalized_odds' in self.fairness_metrics:
                violations['conditional_mmd_equalized_odds'] = self.compute_conditional_mmd_equalized_odds(
                    embeddings, sensitive_attr, labels, kernel_type=self.mmd_kernel
                )
        else:
            # Warn if conditional metrics requested but labels not provided
            if ('conditional_mmd_eod' in self.fairness_metrics or
                'conditional_mmd_equalized_odds' in self.fairness_metrics):
                warnings.warn(
                    "Conditional MMD metrics requested but labels not provided. "
                    "These metrics will be skipped."
                )

        return violations

    def forward(self,
                layer_outputs: List[torch.Tensor],
                sensitive_attr: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                task_loss: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute Progressive Fairness Loss using representation-based fairness

        Args:
            layer_outputs: List of hidden representations from each layer
            sensitive_attr: Sensitive attributes
            labels: Ground truth labels (optional, for compatibility)
            mask: Optional mask for training nodes
            task_loss: Task loss (required for gradient conflict-aware scheduling)

        Returns:
            Dictionary containing total loss and detailed metrics
        """

        # Store original layer outputs for gradient conflict (need full computational graph)
        original_layer_outputs = layer_outputs

        # Apply mask for fairness computation
        if mask is not None:
            sensitive_attr_masked = sensitive_attr[mask]
            layer_outputs_masked = [output[mask] for output in layer_outputs]
            if labels is not None:
                labels = labels[mask]
        else:
            sensitive_attr_masked = sensitive_attr
            layer_outputs_masked = layer_outputs

        total_loss = 0.0
        layer_violations = []
        layer_weights = []
        gradient_conflicts = []

        num_layers = len(layer_outputs)

        # Use gradient conflict-aware scheduling if enabled
        if self.use_gradient_conflict and self.conflict_scheduler is not None:
            if task_loss is None:
                raise ValueError("task_loss must be provided when using gradient conflict-aware scheduling")

            # First, compute total fairness loss to get gradients
            temp_fairness_loss = 0.0
            for embeddings in layer_outputs_masked:
                violations = self.compute_layer_fairness(embeddings, sensitive_attr_masked, labels)
                temp_fairness_loss += sum(violations.values())

            # Compute conflict-aware weights using ORIGINAL (unmasked) layer outputs
            # This preserves the computational graph for gradient calculation
            conflict_weights, conflicts = self.conflict_scheduler.compute_conflict_aware_weights(
                task_loss=task_loss,
                fairness_loss=temp_fairness_loss,
                layer_outputs=original_layer_outputs
            )
            gradient_conflicts = conflicts

            # Now compute weighted loss with conflict-aware weights
            for layer_idx, embeddings in enumerate(layer_outputs_masked):
                weight = conflict_weights[layer_idx]
                layer_weights.append(weight)

                violations = self.compute_layer_fairness(embeddings, sensitive_attr_masked, labels)
                layer_violation = sum(violations.values())
                layer_violations.append(layer_violation.item())

                total_loss += weight * layer_violation

        else:
            # Standard progressive fairness (base schedule only)
            for layer_idx, embeddings in enumerate(layer_outputs_masked):

                # Get progressive weight for this layer
                weight = self.get_layer_weight(layer_idx, num_layers)
                layer_weights.append(weight.item())

                # Compute fairness violations for this layer
                violations = self.compute_layer_fairness(embeddings, sensitive_attr_masked, labels)

                # Aggregate violations for this layer
                layer_violation = sum(violations.values())
                layer_violations.append(layer_violation.item())

                # Add weighted violation to total loss
                total_loss += weight * layer_violation

        # Store for analysis
        self.layer_weights_history.append(layer_weights)
        self.fairness_violations_history.append(layer_violations)
        if gradient_conflicts:
            self.gradient_conflicts_history.append(gradient_conflicts)

        # Prepare detailed return information
        result = {
            'total_loss': total_loss,
            'layer_weights': torch.tensor(layer_weights),
            'layer_violations': torch.tensor(layer_violations),
            'weighted_violations': torch.tensor(layer_weights) * torch.tensor(layer_violations),
            'schedule_type': self.schedule_type,
            'alpha': self.alpha_param.item() if hasattr(self.alpha_param, 'item') else self.alpha_param,
            'beta': self.beta_param.item() if hasattr(self.beta_param, 'item') else self.beta_param,
            'use_gradient_conflict': self.use_gradient_conflict
        }

        # Add gradient conflict information if available
        if gradient_conflicts:
            result['gradient_conflicts'] = torch.tensor(gradient_conflicts)
            result['conflict_strategy'] = self.conflict_scheduler.conflict_strategy if self.conflict_scheduler else None

        return result
