import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score
)
from scipy import stats
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FairnessEvaluator:
    """Comprehensive fairness evaluation framework"""

    def __init__(self):
        self.metrics_history = []

    def calculate_all_metrics(self, predictions, labels, sensitive_attr,
                             probabilities=None, verbose=True):
        """Calculate all fairness and performance metrics"""

        # Convert to numpy for easier computation
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(sensitive_attr):
            sensitive_attr = sensitive_attr.cpu().numpy()
        if probabilities is not None and torch.is_tensor(probabilities):
            probabilities = probabilities.cpu().numpy()

        metrics = {}

        # Performance metrics
        metrics['performance'] = self._calculate_performance_metrics(
            predictions, labels, probabilities
        )

        # Fairness metrics
        metrics['fairness'] = self._calculate_fairness_metrics(
            predictions, labels, sensitive_attr, probabilities
        )

        # Group-specific metrics
        metrics['group_metrics'] = self._calculate_group_metrics(
            predictions, labels, sensitive_attr, probabilities
        )

        
        # Multi-scale specific metrics (if applicable)
        metrics['multiscale'] = self._calculate_multiscale_metrics(
            predictions, labels, sensitive_attr
        )

        if verbose:
            self._print_metrics(metrics)

        self.metrics_history.append(metrics)
        return metrics

    def _calculate_performance_metrics(self, predictions, labels, probabilities=None):
        """Calculate standard performance metrics"""
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'balanced_accuracy': balanced_accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='binary', zero_division=0),
            'recall': recall_score(labels, predictions, average='binary', zero_division=0),
            'f1_score': f1_score(labels, predictions, average='binary', zero_division=0)
        }

        # Per-class metrics
        metrics['precision_per_class'] = precision_score(labels, predictions, average=None, zero_division=0).tolist()
        metrics['recall_per_class'] = recall_score(labels, predictions, average=None, zero_division=0).tolist()
        metrics['f1_per_class'] = f1_score(labels, predictions, average=None, zero_division=0).tolist()

        # Macro and weighted averages
        metrics['precision_macro'] = precision_score(labels, predictions, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(labels, predictions, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(labels, predictions, average='macro', zero_division=0)

        metrics['precision_weighted'] = precision_score(labels, predictions, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(labels, predictions, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(labels, predictions, average='weighted', zero_division=0)

        if probabilities is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(labels, probabilities)
            except:
                metrics['auc_roc'] = np.nan

        
        # Check for prediction collapse (>95% same prediction)
        pred_distribution = np.bincount(predictions.astype(int))
        total_preds = len(predictions)
        for i, count in enumerate(pred_distribution):
            if count > 0.95 * total_preds:
                metrics['warning_prediction_collapse'] = f"Model predicting {count/total_preds:.1%} as class {i}"

        return metrics

    def _calculate_fairness_metrics(self, predictions, labels, sensitive_attr,
                                   probabilities=None):
        """Calculate comprehensive fairness metrics"""

        s0_mask = sensitive_attr == 0
        s1_mask = sensitive_attr == 1

        metrics = {}

        # 1. Statistical Parity Difference (Demographic Parity)
        p_y1_s0 = np.mean(predictions[s0_mask]) if s0_mask.any() else 0
        p_y1_s1 = np.mean(predictions[s1_mask]) if s1_mask.any() else 0
        metrics['statistical_parity_diff'] = abs(p_y1_s0 - p_y1_s1)
        metrics['statistical_parity_ratio'] = p_y1_s1 / p_y1_s0 if p_y1_s0 > 0 else np.inf

        # 2. Equal Opportunity Difference
        positive_labels = labels == 1
        if positive_labels.any():
            tpr_s0 = np.mean(predictions[s0_mask & positive_labels]) if (s0_mask & positive_labels).any() else 0
            tpr_s1 = np.mean(predictions[s1_mask & positive_labels]) if (s1_mask & positive_labels).any() else 0
            metrics['equal_opportunity_diff'] = abs(tpr_s0 - tpr_s1)
            metrics['equal_opportunity_ratio'] = tpr_s1 / tpr_s0 if tpr_s0 > 0 else np.inf
        else:
            metrics['equal_opportunity_diff'] = 0
            metrics['equal_opportunity_ratio'] = 1

        
        return metrics

    def _calculate_group_metrics(self, predictions, labels, sensitive_attr,
                                probabilities=None):
        """Calculate metrics for each sensitive group"""

        groups = {}
        for s in [0, 1]:
            mask = sensitive_attr == s
            if not mask.any():
                continue

            group_preds = predictions[mask]
            group_labels = labels[mask]

            groups[f'group_{s}'] = {
                'size': mask.sum(),
                'positive_rate': np.mean(group_preds),
                'accuracy': accuracy_score(group_labels, group_preds),
                'precision': precision_score(group_labels, group_preds, zero_division=0),
                'recall': recall_score(group_labels, group_preds, zero_division=0),
                'f1_score': f1_score(group_labels, group_preds, zero_division=0)
            }

            if probabilities is not None:
                group_probs = probabilities[mask]
                try:
                    groups[f'group_{s}']['auc_roc'] = roc_auc_score(group_labels, group_probs)
                except:
                    groups[f'group_{s}']['auc_roc'] = np.nan

        return groups


    
    def _print_metrics(self, metrics):
        """Pretty print metrics"""

        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)

        # Performance Metrics
        print("\nPERFORMANCE METRICS:")
        print("-"*30)
        perf = metrics['performance']
        print(f"  Accuracy:  {perf['accuracy']:.4f}")
        print(f"  Precision: {perf['precision']:.4f}")
        print(f"  Recall:    {perf['recall']:.4f}")
        print(f"  F1-Score:  {perf['f1_score']:.4f}")
        if 'auc_roc' in perf:
            print(f"  AUC-ROC:   {perf['auc_roc']:.4f}")

        # Fairness Metrics
        print("\nFAIRNESS METRICS:")
        print("-"*30)
        fair = metrics['fairness']
        print(f"  Statistical Parity Diff:  {fair['statistical_parity_diff']:.4f}")
        print(f"  Equal Opportunity Diff:   {fair['equal_opportunity_diff']:.4f}")
        print(f"  Equalized Odds Diff:      {fair['equalized_odds_diff']:.4f}")
        print(f"  Disparate Impact:         {fair['disparate_impact']:.4f}")

        # Group-specific Metrics
        print("\nGROUP-SPECIFIC METRICS:")
        print("-"*30)
        for group_name, group_metrics in metrics['group_metrics'].items():
            print(f"\n  {group_name.upper()}:")
            print(f"    Size:       {group_metrics['size']}")
            print(f"    Accuracy:   {group_metrics['accuracy']:.4f}")
            print(f"    Pos. Rate:  {group_metrics['positive_rate']:.4f}")

        print("="*60)

    