
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
                 fairness_metrics: List[str] = ['representation_mmd', 'representation_variance'],
                 class_weights: Optional[torch.Tensor] = None,
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
            fairness_metrics: List of representation-based fairness metrics
                             Options: 'representation_mmd', 'representation_variance', 'representation_covariance'
            class_weights: Weights for handling class imbalance
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
        Compute Maximum Mean Discrepancy (MMD) between group embeddings

        This measures if the learned representations are fair - i.e., whether
        embeddings from different sensitive groups come from similar distributions.
        """
        mask_s0 = (sensitive_attr == 0)
        mask_s1 = (sensitive_attr == 1)

        if not mask_s0.any() or not mask_s1.any():
            return torch.tensor(0.0, device=embeddings.device)

        emb_s0 = embeddings[mask_s0]
        emb_s1 = embeddings[mask_s1]

        # Mean embeddings for each group
        mean_s0 = emb_s0.mean(dim=0)
        mean_s1 = emb_s1.mean(dim=0)

        # MMD: L2 distance between mean embeddings
        mmd = torch.norm(mean_s0 - mean_s1, p=2)

        return mmd

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


    def compute_layer_fairness(self,
                              embeddings: torch.Tensor,
                              sensitive_attr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute representation-based fairness violations for a layer

        Args:
            embeddings: Layer embeddings/representations
            sensitive_attr: Sensitive attributes

        Returns:
            Dictionary of fairness violations
        """

        violations = {}

        # Representation-based fairness metrics
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
                violations = self.compute_layer_fairness(embeddings, sensitive_attr_masked)
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

                violations = self.compute_layer_fairness(embeddings, sensitive_attr_masked)
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
                violations = self.compute_layer_fairness(embeddings, sensitive_attr_masked)

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
