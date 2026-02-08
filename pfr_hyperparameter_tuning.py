import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# PFR-specific components
from progressive_fairness_loss import ProgressiveFairnessLoss
from preprocess_data import preprocess_data
from calculate_fairness import calculate_fairness
from utils import set_device
from msfl_evaluation_framework import FairnessEvaluator

# base GNN architecture
from model import GNN




class GNNWithLayerOutputs(nn.Module):
    """
    Wrapper around GNN to extract intermediate layer outputs for PFR
    """
    def __init__(self, base_gnn):
        super().__init__()
        self.gnn = base_gnn

    def forward(self, x, edge_index, return_embeddings=False):
        """
        Forward pass with layer output extraction

        Args:
            x: Node features
            edge_index: Graph structure
            return_embeddings: If True, return (output, embeddings, layer_outputs)

        Returns:
            output, embeddings (if return_embeddings=False)
            output, embeddings, layer_outputs (if return_embeddings=True)
        """
        layer_outputs = []

        # First layer (conv1)
        h = self.gnn.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.gnn.dropout, training=self.training)
        layer_outputs.append(h)  

        # Middle layers (convs)
        for conv in self.gnn.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.gnn.dropout, training=self.training)
            layer_outputs.append(h)  

        # Store embeddings before final layer
        embeddings = h

        # Final layer (conv2)
        h = self.gnn.conv2(h, edge_index)
        layer_outputs.append(h)  

        # Final output (log_softmax)
        output = F.log_softmax(h, dim=1)

        if return_embeddings:
            return output, embeddings, layer_outputs
        else:
            return output, embeddings


class PFRHyperparameterTuning:

    def __init__(self, dataset_name='credit', data_dir='dataset', device=None):
        """
        Initialize PFR hyperparameter tuning framework

        Args:
            dataset_name: Name of dataset ('credit', 'german', 'bail')
            data_dir: Directory containing datasets
            device: Torch device (auto-detected if None)
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.device = device if device else set_device()

        # Load data once
        self.data, self.sensitive_attr = preprocess_data(
            data_dir, dataset_name, train_split=0.6, test_split=0.2
        )
        self.data = self.data.to(self.device)
        self.sensitive_attr = self.sensitive_attr.to(self.device)
        self.data.sensitive_attr = self.sensitive_attr

        # Determine output dimension
        self.num_classes = len(torch.unique(self.data.y))
        self.output_dim = self.num_classes

        # Calculate class weights for handling imbalance
        train_labels = self.data.y[self.data.train_mask]
        class_counts = torch.bincount(train_labels.long())
        total_samples = len(train_labels)

        # Calculate weights: inverse frequency weighted by total samples
        self.class_weights = total_samples / (self.num_classes * class_counts.float())
        self.class_weights = self.class_weights.to(self.device)

        # Print class distribution info
        print(f"\nClass distribution in training set:")
        for i in range(self.num_classes):
            pct = (class_counts[i].float() / total_samples * 100).item()
            print(f"  Class {i}: {class_counts[i]} samples ({pct:.1f}%), weight: {self.class_weights[i]:.3f}")

        # Calculate imbalance ratio
        imbalance_ratio = class_counts.max().float() / class_counts.min().float()
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1\n")

        # Initialize evaluator
        self.evaluator = FairnessEvaluator()

        # Define search spaces for different model types
        self.define_search_spaces()

        # Results storage
        self.results = {
            'dataset': dataset_name,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
            'trials': []
        }

    def define_search_spaces(self):
        """Define hyperparameter search spaces for all model types"""

        # Common neural network hyperparameters
        self.common_space = {
            'hidden_dim': [32, 64, 128],
            'num_layers': [2, 3, 4],
            'dropout': [0.3, 0.5, 0.7],
            'learning_rate': [0.001, 0.005, 0.01, 0.05],
            'weight_decay': [0.0, 5e-4, 1e-3],
            'epochs': [200, 300]
        }

        # PFR specific parameters
        self.pfr_space = {
            'schedule_type': ['linear', 'exponential', 'sigmoid'],
            'alpha': [0.5, 1.0, 2.0],
            'beta': [0.0, 0.1, 0.2],
            'lambda_pfr': [0.1, 0.3, 0.5, 1.0],
            'fairness_metrics': [
                ['representation_mmd'],
                ['conditional_mmd_eod'],
                ['representation_mmd', 'conditional_mmd_eod'],
                ['representation_mmd', 'representation_variance'],
                ['conditional_mmd_equalized_odds']
            ],
            'mmd_kernel': ['rbf', 'linear'],
            'kernel_num': [3, 5, 7]
        }

        # Gradient Conflict specific parameters
        self.gc_space = {
            'conflict_strategy': ['inverse', 'threshold', 'exponential'],
            'conflict_threshold': [-0.3, -0.2, -0.1],
            'conflict_smoothing': [0.8, 0.9],
            'use_smoothed_conflicts': [True, False]
        }

    def create_model_and_loss(self, model_type: str, params: Dict) -> Tuple[nn.Module, Any]:
        """
        Create model and loss function based on type and parameters

        Args:
            model_type: Type of model ('vanilla', 'pfr', 'gc_pfr')
            params: Hyperparameters dictionary

        Returns:
            model, loss_function tuple
        """
        # Create base GNN model
        base_model = GNN(
            data=self.data,
            model=params.get('gnn_type', 'GCN'),
            fame=False,
            enhanced=False,
            sens_attribute=self.sensitive_attr,
            layers=params['num_layers'],
            hidden=params['hidden_dim'],
            dropout=params['dropout']
        )

        # Wrap to extract layer outputs
        model = GNNWithLayerOutputs(base_model).to(self.device)

        # Create loss function based on model type
        if model_type == 'vanilla':
            # No fairness - NLLLoss with class weights
            loss_fn = nn.NLLLoss(weight=self.class_weights)

        elif model_type in ['pfr', 'gc_pfr']:
            # Progressive Fairness Regularization
            use_gc = (model_type == 'gc_pfr')

            loss_fn = ProgressiveFairnessLoss(
                schedule_type=params.get('schedule_type', 'linear'),
                alpha=params.get('alpha', 1.0),
                beta=params.get('beta', 0.1),
                learnable_schedule=False,
                fairness_metrics=params.get('fairness_metrics', ['representation_mmd', 'conditional_mmd_eod']),
                class_weights=None, 
                # MMD parameters
                mmd_kernel=params.get('mmd_kernel', 'rbf'),
                kernel_num=params.get('kernel_num', 5),
                # Gradient conflict parameters
                use_gradient_conflict=use_gc,
                conflict_strategy=params.get('conflict_strategy', 'inverse') if use_gc else 'inverse',
                conflict_threshold=params.get('conflict_threshold', -0.1) if use_gc else -0.1,
                conflict_smoothing=params.get('conflict_smoothing', 0.9) if use_gc else 0.9,
                use_smoothed_conflicts=params.get('use_smoothed_conflicts', True) if use_gc else True
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model, loss_fn

    def train_and_evaluate(self, model: nn.Module, loss_fn: Any, params: Dict, model_type: str) -> Dict:
        """
        Train model and evaluate performance

        Args:
            model: The GNN model (wrapped with layer extraction)
            loss_fn: Loss function (PFR or vanilla)
            params: Training hyperparameters
            model_type: Type of model ('vanilla', 'pfr', 'gc_pfr')

        Returns:
            Dictionary with evaluation metrics
        """

        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params.get('weight_decay', 0)
        )

        # Training settings
        epochs = params.get('epochs', 200)
        patience = 30
        best_val_acc = 0
        patience_counter = 0

        lambda_pfr = params.get('lambda_pfr', 0.5) if model_type != 'vanilla' else 0.0

        # Training loop
        model.train()

        # Track correlation every 20 epochs for PFR/GC-PFR
        track_correlation_interval = 20

        # Store epoch-level metrics for analysis
        epoch_history = {
            'epoch': [],
            'train_loss': [],
            'val_acc': [],
            'mmd_spd_correlation': [],
            'conditional_mmd_eod_correlation': [],
            'latest_mmd': [],
            'latest_spd': [],
            'latest_eod': []
        }

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass with layer extraction
            output, embeddings, layer_outputs = model(self.data.x, self.data.edge_index, return_embeddings=True)

            # Get predictions for training nodes
            train_preds = output[self.data.train_mask]
            train_labels = self.data.y[self.data.train_mask]
            train_sensitive = self.sensitive_attr[self.data.train_mask]

            # Compute task loss (with class weights)
            task_loss = F.nll_loss(train_preds, train_labels.long(), weight=self.class_weights)

            # Compute total loss
            if model_type == 'vanilla':
                loss = task_loss
            else:
                # PFR or GC-PFR
                pfr_result = loss_fn(
                    layer_outputs=layer_outputs,
                    sensitive_attr=self.sensitive_attr,
                    labels=self.data.y,
                    mask=self.data.train_mask,
                    task_loss=task_loss  # Required for gradient conflict
                )
                pfr_loss = pfr_result['total_loss']

                # Combined loss
                loss = task_loss + lambda_pfr * pfr_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Store training loss
            epoch_history['epoch'].append(epoch)
            epoch_history['train_loss'].append(loss.item())

            # Validation for early stopping and correlation tracking
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_output, val_embeddings, _ = model(self.data.x, self.data.edge_index, return_embeddings=True)
                    val_preds = val_output[self.data.val_mask]
                    val_labels = self.data.y[self.data.val_mask]

                    val_acc = (val_preds.argmax(dim=1) == val_labels).float().mean().item()

                    # Store validation accuracy at this epoch
                    epoch_history['val_acc'].append(val_acc)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Track correlation for PFR/GC-PFR models
                    if model_type != 'vanilla' and epoch % track_correlation_interval == 0:
                        # Get predictions as probabilities
                        val_probs = torch.exp(val_output[self.data.val_mask])[:, 1] if val_output.size(1) > 1 else torch.exp(val_output[self.data.val_mask])[:, 0]

                        loss_fn.update_correlation_metrics(
                            embeddings=val_embeddings[self.data.val_mask],
                            predictions=val_probs,
                            labels=self.data.y[self.data.val_mask],
                            sensitive_attr=self.sensitive_attr[self.data.val_mask]
                        )

                        # Get correlation statistics at this epoch
                        corr_stats = loss_fn.get_correlation_statistics()
                        epoch_history['mmd_spd_correlation'].append(
                            corr_stats.get('mmd_spd_correlation', np.nan)
                        )
                        epoch_history['conditional_mmd_eod_correlation'].append(
                            corr_stats.get('conditional_mmd_eod_correlation', np.nan)
                        )
                        epoch_history['latest_mmd'].append(
                            corr_stats.get('latest_mmd', np.nan)
                        )
                        epoch_history['latest_spd'].append(
                            corr_stats.get('latest_spd', np.nan)
                        )
                        epoch_history['latest_eod'].append(
                            corr_stats.get('latest_eod', np.nan)
                        )

                    if patience_counter >= patience:
                        break

                model.train()

        # Final evaluation on test set
        model.eval()
        with torch.no_grad():
            test_output, test_embeddings, test_layer_outputs = model(
                self.data.x, self.data.edge_index, return_embeddings=True
            )
            test_preds = test_output[self.data.test_mask]
            test_labels = self.data.y[self.data.test_mask]
            test_sensitive = self.sensitive_attr[self.data.test_mask]

            # Get predictions from log_softmax output
            test_pred_classes = test_preds.argmax(dim=1)
            # Convert log probabilities to probabilities for positive class
            test_probs = torch.exp(test_preds)[:, 1] if test_preds.size(1) > 1 else torch.exp(test_preds)[:, 0]

            # Calculate standard metrics
            metrics = self.evaluator.calculate_all_metrics(
                test_pred_classes, test_labels, test_sensitive, test_probs, verbose=False
            )

            # Add training info
            metrics['epochs_trained'] = epoch + 1
            metrics['best_val_acc'] = best_val_acc

            # Add epoch-level history
            metrics['epoch_history'] = epoch_history

            # Add PFR-specific metrics if applicable
            if model_type != 'vanilla':
                # Compute PFR metrics on test set
                test_task_loss = F.nll_loss(test_preds, test_labels.long(), weight=self.class_weights)
                pfr_result = loss_fn(
                    layer_outputs=test_layer_outputs,
                    sensitive_attr=self.sensitive_attr,
                    labels=self.data.y,
                    mask=self.data.test_mask,
                    task_loss=test_task_loss
                )

                metrics['pfr_specific'] = {
                    'total_pfr_loss': pfr_result['total_loss'].item(),
                    'layer_weights': pfr_result['layer_weights'].cpu().numpy().tolist(),
                    'layer_violations': pfr_result['layer_violations'].cpu().numpy().tolist(),
                    'weighted_violations': pfr_result['weighted_violations'].cpu().numpy().tolist(),
                    'schedule_type': pfr_result['schedule_type'],
                    'alpha': pfr_result['alpha'],
                    'beta': pfr_result['beta'],
                    'lambda_pfr': lambda_pfr,
                    'fairness_metrics_used': params.get('fairness_metrics', []),
                    'mmd_kernel': params.get('mmd_kernel', 'rbf'),
                    'kernel_num': params.get('kernel_num', 5)
                }

                # Add correlation statistics
                correlation_stats = loss_fn.get_correlation_statistics()
                metrics['pfr_specific']['correlation_statistics'] = correlation_stats

                # Add gradient conflict info if GC-PFR
                if 'gradient_conflicts' in pfr_result:
                    metrics['pfr_specific']['gradient_conflicts'] = pfr_result['gradient_conflicts'].cpu().numpy().tolist()
                    metrics['pfr_specific']['conflict_strategy'] = pfr_result['conflict_strategy']
                    metrics['pfr_specific']['use_gradient_conflict'] = True
                else:
                    metrics['pfr_specific']['use_gradient_conflict'] = False

        return metrics

    def objective_optuna(self, trial: optuna.Trial, model_type: str) -> float:
        """
        Optuna objective function for hyperparameter optimization

        Args:
            trial: Optuna trial object
            model_type: Type of model to optimize ('vanilla', 'pfr', 'gc_pfr')

        Returns:
            Objective value (higher is better)
        """

        # Sample common hyperparameters
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.3, 0.7, step=0.1),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 5e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
            'epochs': 300  # Fixed with early stopping
        }

        # Add model-specific parameters
        if model_type in ['pfr', 'gc_pfr']:
            params['schedule_type'] = trial.suggest_categorical('schedule_type', ['linear', 'exponential', 'sigmoid'])
            params['alpha'] = trial.suggest_categorical('alpha', [0.5, 1.0, 2.0])
            params['beta'] = trial.suggest_categorical('beta', [0.0, 0.1, 0.2])
            params['lambda_pfr'] = trial.suggest_categorical('lambda_pfr', [0.1, 0.3, 0.5, 1.0])

            # Fairness metrics 
            metrics_options = [
                ['representation_mmd'],
                ['conditional_mmd_eod'],
                ['representation_mmd', 'conditional_mmd_eod'],
                ['representation_mmd', 'representation_variance'],
                ['conditional_mmd_equalized_odds']
            ]
            metrics_idx = trial.suggest_categorical('fairness_metrics_idx', [0, 1, 2, 3, 4])
            params['fairness_metrics'] = metrics_options[metrics_idx]

            # MMD kernel parameters
            params['mmd_kernel'] = trial.suggest_categorical('mmd_kernel', ['rbf', 'linear'])
            params['kernel_num'] = trial.suggest_categorical('kernel_num', [3, 5, 7])

        # Add gradient conflict parameters for gc_pfr
        if model_type == 'gc_pfr':
            params['conflict_strategy'] = trial.suggest_categorical('conflict_strategy', ['inverse', 'threshold', 'exponential'])
            params['conflict_threshold'] = trial.suggest_categorical('conflict_threshold', [-0.3, -0.2, -0.1])
            params['conflict_smoothing'] = trial.suggest_categorical('conflict_smoothing', [0.8, 0.9])
            params['use_smoothed_conflicts'] = trial.suggest_categorical('use_smoothed_conflicts', [True, False])

        # Create model and loss
        try:
            model, loss_fn = self.create_model_and_loss(model_type, params)

            # Train and evaluate
            metrics = self.train_and_evaluate(model, loss_fn, params, model_type)

            # Composite objective: balance accuracy and fairness
            accuracy = metrics['performance']['accuracy']
            balanced_acc = metrics['performance'].get('balanced_accuracy', accuracy)
            spd = metrics['fairness']['statistical_parity_diff']
            eod = metrics['fairness']['equal_opportunity_diff']

            # Objective: maximize balanced accuracy while minimizing unfairness
            objective = balanced_acc - 0.3 * spd - 0.2 * eod

            # Store trial results with ALL metrics
            trial_result = {
                'trial': trial.number,
                'model_type': model_type,
                'params': params,
                'metrics': metrics,
                'objective': objective,
                'summary': {
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'auc_roc': metrics['performance'].get('auc_roc', 0),
                    'f1_macro': metrics['performance'].get('f1_macro', 0),
                    'spd': spd,
                    'eod': eod
                }
            }
            self.results['trials'].append(trial_result)

            # Save intermediate results after each trial
            self.save_trial_results(trial.number, model_type, trial_result)

            # Print detailed metrics
            print(f"\n{'='*70}")
            print(f"Trial {trial.number} Results ({model_type})")
            print(f"{'='*70}")

            print(f"\n  ===== Performance Metrics =====")
            print(f"  Accuracy: {accuracy:.4f} | Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  AUC-ROC: {metrics['performance'].get('auc_roc', 0):.4f}")
            print(f"  F1 Score (macro): {metrics['performance'].get('f1_macro', 0):.4f}")
            print(f"  F1 Score (weighted): {metrics['performance'].get('f1_weighted', 0):.4f}")

            # Per-class metrics
            prec_per_class = metrics['performance'].get('precision_per_class', [])
            rec_per_class = metrics['performance'].get('recall_per_class', [])
            f1_per_class = metrics['performance'].get('f1_per_class', [])

            for i in range(len(prec_per_class)):
                print(f"  Class {i}: Prec={prec_per_class[i]:.3f}, Rec={rec_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")

            print(f"\n  ===== Fairness Metrics =====")
            print(f"  Statistical Parity Diff: {spd:.4f}")
            print(f"  Equal Opportunity Diff: {eod:.4f}")

            # PFR-specific metrics
            if 'pfr_specific' in metrics:
                pfr = metrics['pfr_specific']
                print(f"\n  ===== PFR-Specific Metrics =====")
                print(f"  Schedule: {pfr['schedule_type']} (α={pfr['alpha']}, β={pfr['beta']}), λ_PFR={pfr['lambda_pfr']}")
                print(f"  Fairness Metrics: {pfr['fairness_metrics_used']}")
                print(f"  MMD Kernel: {pfr.get('mmd_kernel', 'rbf')} (num_kernels={pfr.get('kernel_num', 5)})")

                if pfr.get('use_gradient_conflict', False):
                    print(f"  Gradient Conflict: True (strategy={pfr['conflict_strategy']})")
                else:
                    print(f"  Gradient Conflict: False")

                layer_weights = pfr['layer_weights']
                layer_violations = pfr['layer_violations']
                print(f"  Layer Weights: [{', '.join([f'{w:.3f}' for w in layer_weights])}]")
                print(f"  Layer Violations: [{', '.join([f'{v:.3f}' for v in layer_violations])}]")

                if 'gradient_conflicts' in pfr:
                    conflicts = pfr['gradient_conflicts']
                    print(f"  Gradient Conflicts: [{', '.join([f'{c:.3f}' for c in conflicts])}]")

                # Display correlation statistics
                if 'correlation_statistics' in pfr:
                    corr_stats = pfr['correlation_statistics']
                    print(f"\n  ===== Correlation Statistics =====")
                    if 'mmd_spd_correlation' in corr_stats:
                        corr_val = corr_stats['mmd_spd_correlation']
                        status = "✓ STRONG" if abs(corr_val) > 0.6 else ("⚠ MODERATE" if abs(corr_val) > 0.3 else "✗ WEAK")
                        print(f"  MMD ↔ SPD Correlation:  {corr_val:+.4f}  {status}")
                    if 'conditional_mmd_eod_correlation' in corr_stats:
                        corr_val = corr_stats['conditional_mmd_eod_correlation']
                        status = "✓ STRONG" if abs(corr_val) > 0.6 else ("⚠ MODERATE" if abs(corr_val) > 0.3 else "✗ WEAK")
                        print(f"  Conditional MMD ↔ EOD:  {corr_val:+.4f}  {status}")
                    if 'note' not in corr_stats and 'latest_mmd' in corr_stats:
                        print(f"  Latest MMD: {corr_stats.get('latest_mmd', 0):.6f}, SPD: {corr_stats.get('latest_spd', 0):.4f}")
                        if 'latest_conditional_mmd' in corr_stats:
                            print(f"  Latest Cond MMD: {corr_stats.get('latest_conditional_mmd', 0):.6f}, EOD: {corr_stats.get('latest_eod', 0):.4f}")

            print(f"\n  Objective Score: {objective:.4f}")
            print(f"{'='*70}\n")

            # Warning if model is collapsing
            if 'warning_prediction_collapse' in metrics['performance']:
                print(f"    WARNING: {metrics['performance']['warning_prediction_collapse']}")

            return objective

        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0

    def optuna_search(self, model_type: str, n_trials: int = 50, timeout: int = None) -> optuna.Study:
        """
        Perform Optuna-based hyperparameter optimization

        Args:
            model_type: Type of model to optimize ('vanilla', 'pfr', 'gc_pfr')
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (optional)

        Returns:
            Optuna study object
        """

        print(f"\n{'='*70}")
        print(f"Starting Optuna optimization for {model_type.upper()}")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Number of trials: {n_trials}")
        print(f"{'='*70}\n")

        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"{model_type}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Optimize
        study.optimize(
            lambda trial: self.objective_optuna(trial, model_type),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1
        )

        # Print results
        print("\n" + "="*70)
        print(f"OPTIMIZATION COMPLETE: {model_type.upper()}")
        print("="*70)
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"\nBest trial:")
        print(f"  Value: {study.best_value:.4f}")
        print(f"  Params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        print("="*70 + "\n")

        return study

    def save_trial_results(self, trial_number: int, model_type: str, trial_result: dict):
        """Save results after each individual trial"""

        # Create results directory if it doesn't exist
        os.makedirs('results/trials', exist_ok=True)

        # Save this trial's results
        filename = f"results/trials/{self.dataset_name}_{model_type}_trial_{trial_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(trial_result, f, indent=2, default=str)

        # Also update the cumulative results file
        cumulative_filename = f"results/trials/{self.dataset_name}_{model_type}_all_trials.json"
        with open(cumulative_filename, 'w') as f:
            json.dump({
                'dataset': self.dataset_name,
                'model_type': model_type,
                'timestamp': datetime.now().isoformat(),
                'num_trials_completed': len(self.results['trials']),
                'trials': self.results['trials']
            }, f, indent=2, default=str)

    def save_results(self, comparison_df: pd.DataFrame, filename: str = None):
        """Save tuning results to files"""

        if filename is None:
            filename = f"pfr_hyperparameter_tuning_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

        # Save as CSV
        csv_path = f"results/{filename}.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Save detailed results as JSON
        json_path = f"results/{filename}_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Detailed results saved to {json_path}")

        # Generate report
        self.generate_report(comparison_df, filename)

    def generate_report(self, comparison_df: pd.DataFrame, filename: str):
        """Generate a text report of the results"""

        report_path = f"results/{filename}_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PFR HYPERPARAMETER TUNING REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write("BEST MODEL COMPARISON:\n")
            f.write("-"*40 + "\n")

            for _, row in comparison_df.iterrows():
                f.write(f"\nModel: {row['model_type']}\n")
                f.write(f"  Accuracy: {row['accuracy']:.4f}\n")
                f.write(f"  Balanced Accuracy: {row.get('balanced_accuracy', 0):.4f}\n")
                f.write(f"  SPD: {row['spd']:.4f}\n")
                f.write(f"  EOD: {row['eod']:.4f}\n")
                f.write(f"  Objective: {row['objective']:.4f}\n")

                if isinstance(row.get('best_params'), dict):
                    f.write("  Best Parameters:\n")
                    for key, value in row['best_params'].items():
                        f.write(f"    {key}: {value}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("WINNER: " + comparison_df.iloc[0]['model_type'] + "\n")
            f.write("="*80 + "\n")

        print(f"Report saved to {report_path}")


def main():
    """Main function for PFR hyperparameter tuning"""

    import argparse

    parser = argparse.ArgumentParser(description='PFR Hyperparameter Tuning')
    parser.add_argument('--dataset', type=str, default='credit',
                       choices=['credit', 'german', 'bail'],
                       help='Dataset to use')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['vanilla', 'pfr', 'gc_pfr'],
                       help='Specific model to tune (if None, tunes all)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer trials')
    args = parser.parse_args()

    # Initialize tuner
    tuner = PFRHyperparameterTuning(
        dataset_name=args.dataset,
        data_dir='dataset'
    )

    if args.quick:
        args.n_trials = 10

    try:
        if args.model_type:
            # Tune specific model
            study = tuner.optuna_search(args.model_type, n_trials=args.n_trials)
        else:
            # Tune all models
            print("\nTuning all model types: pfr, gc_pfr\n")

            studies = {}
            for model_type in ['pfr', 'gc_pfr']:
                print(f"\n{'#'*70}")
                print(f"# Starting optimization for: {model_type.upper()}")
                print(f"{'#'*70}\n")

                studies[model_type] = tuner.optuna_search(model_type, n_trials=args.n_trials)

            # Create comparison
            comparison_data = []
            for model_type, study in studies.items():
                best_trial = study.best_trial
                comparison_data.append({
                    'model_type': model_type,
                    'objective': study.best_value,
                    'accuracy': best_trial.user_attrs.get('accuracy', 0),
                    'balanced_accuracy': best_trial.user_attrs.get('balanced_accuracy', 0),
                    'spd': best_trial.user_attrs.get('spd', 0),
                    'eod': best_trial.user_attrs.get('eod', 0),
                    'best_params': study.best_params
                })

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('objective', ascending=False)

            # Save results
            tuner.save_results(comparison_df)

            # Print final comparison
            print("\n" + "="*80)
            print("FINAL COMPARISON")
            print("="*80)
            print(comparison_df.to_string(index=False))
            print("="*80)

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print("Partial results have been saved.")
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nPFR hyperparameter tuning complete!")


if __name__ == "__main__":
    main()
