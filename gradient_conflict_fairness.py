import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class GradientConflictAnalyzer(nn.Module):
    """
    Analyzes gradient conflicts between task and fairness objectives

    Computes cosine similarity between task and fairness gradients at each layer:
    - cos < 0: Conflicting (gradients point in opposite directions)
    - cos = 0: Orthogonal (independent objectives)
    - cos > 0: Aligned (objectives agree)
    """

    def __init__(self,
                 conflict_threshold: float = -0.1,
                 smoothing_factor: float = 0.9):
        """
        Initialize Gradient Conflict Analyzer

        Args:
            conflict_threshold: Threshold below which conflicts are considered severe
            smoothing_factor: Exponential smoothing factor for conflict history (0-1)
        """
        super().__init__()

        self.conflict_threshold = conflict_threshold
        self.smoothing_factor = smoothing_factor

        # Track conflict history
        self.conflict_history = []
        self.smoothed_conflicts = None

    def compute_gradient_conflict(self,
                                  task_loss: torch.Tensor,
                                  fairness_loss: torch.Tensor,
                                  layer_representation: torch.Tensor) -> float:
        """
        Compute gradient conflict for a single layer

        Args:
            task_loss: Task loss (e.g., cross-entropy)
            fairness_loss: Fairness loss
            layer_representation: Hidden representation at this layer

        Returns:
            Conflict score (cosine similarity): [-1, 1]
        """

        if not layer_representation.requires_grad:
            layer_representation.requires_grad_(True)

        # Compute task gradient
        grad_task = torch.autograd.grad(
            task_loss,
            layer_representation,
            retain_graph=True,
            create_graph=False
        )[0]

        # Compute fairness gradient
        grad_fair = torch.autograd.grad(
            fairness_loss,
            layer_representation,
            retain_graph=True,
            create_graph=False
        )[0]

        # Flatten gradients
        grad_task_flat = grad_task.flatten()
        grad_fair_flat = grad_fair.flatten()

        # Cosine similarity (conflict measure)
        cos_sim = F.cosine_similarity(
            grad_task_flat.unsqueeze(0),
            grad_fair_flat.unsqueeze(0),
            dim=1
        ).item()

        return cos_sim

    def compute_layer_conflicts(self,
                                task_loss: torch.Tensor,
                                fairness_loss: torch.Tensor,
                                layer_outputs: List[torch.Tensor]) -> List[float]:
        """
        Compute gradient conflicts for all layers

        Args:
            task_loss: Task objective loss
            fairness_loss: Fairness objective loss
            layer_outputs: List of layer representations

        Returns:
            List of conflict scores, one per layer
        """

        conflicts = []

        for layer_idx, h_l in enumerate(layer_outputs):
            try:
                conflict = self.compute_gradient_conflict(
                    task_loss, fairness_loss, h_l
                )
                conflicts.append(conflict)
            except Exception as e:
                # If gradient computation fails, assume neutral
                warnings.warn(f"Failed to compute conflict for layer {layer_idx}: {e}")
                conflicts.append(0.0)

        # Update history
        self.conflict_history.append(conflicts)

        # Exponential smoothing
        if self.smoothed_conflicts is None:
            self.smoothed_conflicts = conflicts.copy()
        else:
            self.smoothed_conflicts = [
                self.smoothing_factor * old + (1 - self.smoothing_factor) * new
                for old, new in zip(self.smoothed_conflicts, conflicts)
            ]

        return conflicts

    def conflict_to_weight_modifier(self,
                                   conflict_score: float,
                                   strategy: str = 'inverse') -> float:
        """
        Convert conflict score to weight modifier

        Args:
            conflict_score: Cosine similarity between gradients [-1, 1]
            strategy: How to convert conflict to weight
                     'inverse': More weight for conflicts
                     'threshold': Binary based on threshold
                     'exponential': Exponential scaling

        Returns:
            Weight modifier [0, 2+]
        """

        if strategy == 'inverse':
            # More weight where gradients conflict (negative cosine)
            # Range: [1.0, 2.0] for conflicts, [0.5, 1.0] for alignment
            modifier = max(0, -conflict_score) + 1.0

        elif strategy == 'threshold':
            # Binary: high weight if conflict, low if aligned
            if conflict_score < self.conflict_threshold:
                modifier = 2.0
            else:
                modifier = 1.0

        elif strategy == 'exponential':
            # Exponential scaling based on conflict severity
            modifier = torch.exp(torch.tensor(-conflict_score)).item()
            # Clip to reasonable range
            modifier = max(0.5, min(modifier, 3.0))

        else:
            modifier = 1.0

        return modifier

    def get_conflict_statistics(self) -> Dict[str, any]:
        """Get statistics about observed conflicts"""

        if not self.conflict_history:
            return {}

        # Convert to numpy for analysis
        conflicts_array = np.array(self.conflict_history)  # [num_steps, num_layers]

        stats = {
            'num_observations': len(self.conflict_history),
            'num_layers': conflicts_array.shape[1] if len(conflicts_array.shape) > 1 else 0,
            'per_layer_stats': []
        }

        if len(conflicts_array.shape) > 1:
            for layer_idx in range(conflicts_array.shape[1]):
                layer_conflicts = conflicts_array[:, layer_idx]
                stats['per_layer_stats'].append({
                    'layer': layer_idx,
                    'mean_conflict': float(np.mean(layer_conflicts)),
                    'std_conflict': float(np.std(layer_conflicts)),
                    'min_conflict': float(np.min(layer_conflicts)),
                    'max_conflict': float(np.max(layer_conflicts)),
                    'severe_conflicts': int(np.sum(layer_conflicts < self.conflict_threshold))
                })

        return stats


class ConflictAwareWeightScheduler(nn.Module):
    """
    Schedules fairness weights based on gradient conflicts

    Combines base progressive schedule with conflict-based adaptation:
    w(l) = base_schedule(l) × conflict_modifier(l)
    """

    def __init__(self,
                 base_schedule_type: str = 'linear',
                 alpha: float = 1.0,
                 beta: float = 0.1,
                 conflict_strategy: str = 'inverse',
                 use_smoothed_conflicts: bool = True):
        """
        Initialize Conflict-Aware Weight Scheduler

        Args:
            base_schedule_type: Base progressive schedule ('linear', 'exponential', etc.)
            alpha: Schedule parameter
            beta: Schedule parameter
            conflict_strategy: How to convert conflicts to weights
            use_smoothed_conflicts: Whether to use smoothed conflict estimates
        """
        super().__init__()

        self.base_schedule_type = base_schedule_type
        self.alpha = alpha
        self.beta = beta
        self.conflict_strategy = conflict_strategy
        self.use_smoothed_conflicts = use_smoothed_conflicts

        # Conflict analyzer
        self.conflict_analyzer = GradientConflictAnalyzer()

    def get_base_weight(self, layer_idx: int, total_layers: int) -> float:
        """Compute base progressive weight"""

        l = layer_idx + 1  # 1-based indexing

        if self.base_schedule_type == 'linear':
            weight = self.alpha / l + self.beta
        elif self.base_schedule_type == 'exponential':
            weight = self.alpha * np.exp(-self.beta * l)
        elif self.base_schedule_type == 'power':
            weight = self.alpha * (l ** (-self.beta))
        else:
            weight = 1.0

        return max(weight, 1e-6)

    def compute_conflict_aware_weights(self,
                                      task_loss: torch.Tensor,
                                      fairness_loss: torch.Tensor,
                                      layer_outputs: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
        """
        Compute conflict-aware weights for all layers

        Args:
            task_loss: Task objective
            fairness_loss: Fairness objective
            layer_outputs: Layer representations

        Returns:
            Tuple of (final_weights, conflict_scores)
        """

        num_layers = len(layer_outputs)

        # Compute conflicts
        conflicts = self.conflict_analyzer.compute_layer_conflicts(
            task_loss, fairness_loss, layer_outputs
        )

        # Use smoothed conflicts if requested
        if self.use_smoothed_conflicts and self.conflict_analyzer.smoothed_conflicts:
            conflicts = self.conflict_analyzer.smoothed_conflicts

        # Compute final weights
        final_weights = []
        for layer_idx in range(num_layers):
            # Base progressive weight
            base_weight = self.get_base_weight(layer_idx, num_layers)

            # Conflict modifier
            conflict_modifier = self.conflict_analyzer.conflict_to_weight_modifier(
                conflicts[layer_idx],
                strategy=self.conflict_strategy
            )

            # Combined weight
            final_weight = base_weight * conflict_modifier
            final_weights.append(final_weight)

        return final_weights, conflicts