import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple
import itertools
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Import MSFL-specific components
from msfl_loss import MultiScaleFairnessLoss
from preprocess_data import preprocess_data
from calculate_fairness import calculate_fairness
from utils import set_device
from msfl_evaluation_framework import FairnessEvaluator

# EXACT SAME GNN architecture from model.py 
from model import GNN
class MSFLHyperparameterTuning:
    
    def __init__(self, dataset_name='credit', data_dir='dataset', device=None):
        """
        Initialize hyperparameter tuning framework

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
        # Always use num_classes for output dimension to match model architectures
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
            'weight_decay': [0.0, 5e-4, 1e-3, 5e-3],
            'epochs': [200, 300, 500]  # Will use early stopping
        }


        # MSFL specific
        self.msfl_space = {
            'lambda_node': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'lambda_neighbor': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'lambda_graph': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'lambda_accuracy': [0.5, 0.8, 1.0, 1.2],  # Task weight
            'similarity_threshold': [0.6, 0.7, 0.8, 0.9],
            'neighborhood_hops': [1, 2, 3]
        }

       
    def create_model_and_loss(self, model_type: str, params: Dict) -> Tuple[nn.Module, Any]:
        """
        Create model and loss function based on type and parameters

        Args:
            model_type: Type of model ('msfl')
            params: Hyperparameters dictionary

        Returns:
            model, loss_function tuple
        """
        # Create base model based on type
        model = GNN(
            data=self.data,
            model=params.get('model', 'GCN'),
            fame=False,  # No FAME for MSFL experiments
            enhanced=False,
            sens_attribute=self.sensitive_attr,
            layers=params['num_layers'],
            hidden=params['hidden_dim'],
            dropout=params['dropout']
        ).to(self.device)
        
        loss_fn = MultiScaleFairnessLoss(
            lambda_node=params.get('lambda_node', 0.3),
            lambda_neighbor=params.get('lambda_neighbor', 0.3),
            lambda_graph=params.get('lambda_graph', 0.4),
            lambda_accuracy=params.get('lambda_accuracy', 1.0),
            similarity_threshold=params.get('similarity_threshold', 0.8),
            neighborhood_hops=params.get('neighborhood_hops', 2),
            class_weights=self.class_weights  # Pass class weights for handling imbalance
        )

        
        return model, loss_fn

    def train_and_evaluate(self, model: nn.Module, loss_fn: Any, params: Dict) -> Dict:
        """
        Train model and evaluate performance

        Args:
            model: The GNN model
            loss_fn: Loss function
            params: Training hyperparameters

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

        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            output, embeddings = model(self.data.x, self.data.edge_index, return_embeddings=True)

            # Get predictions for training nodes
            train_preds = output[self.data.train_mask]
            train_labels = self.data.y[self.data.train_mask]
            train_sensitive = self.sensitive_attr[self.data.train_mask]

            # Compute loss
            if isinstance(loss_fn, (MultiScaleFairnessLoss)):
                
                # The loss function needs access to all nodes for neighborhood calculations
                loss_dict = loss_fn(
                    output, self.data.y, self.sensitive_attr,
                    embeddings, self.data.edge_index,
                    mask=self.data.train_mask  
                )
                loss = loss_dict['total_loss']
            else:
                # For NLLLoss with log_softmax output
                loss = loss_fn(train_preds, train_labels.long())

            # Backward pass
            loss.backward()
            optimizer.step()

            # Validation for early stopping 
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_output, _ = model(self.data.x, self.data.edge_index, return_embeddings=True)
                    val_preds = val_output[self.data.val_mask]
                    val_labels = self.data.y[self.data.val_mask]

                    # For log_softmax output, use argmax
                    val_acc = (val_preds.argmax(dim=1) == val_labels).float().mean().item()

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        break

                model.train()

        # Final evaluation on test set
        model.eval()
        with torch.no_grad():
            test_output, test_embeddings = model(self.data.x, self.data.edge_index, return_embeddings=True)
            test_preds = test_output[self.data.test_mask]
            test_labels = self.data.y[self.data.test_mask]
            test_sensitive = self.sensitive_attr[self.data.test_mask]

            # Get predictions from log_softmax output
            test_pred_classes = test_preds.argmax(dim=1)
            # Convert log probabilities to probabilities for positive class
            test_probs = torch.exp(test_preds)[:, 1] if test_preds.size(1) > 1 else torch.exp(test_preds)[:, 0]

            # Calculate metrics - Set verbose=False for speed
            metrics = self.evaluator.calculate_all_metrics(
                test_pred_classes, test_labels, test_sensitive, test_probs, verbose=True
            )

            # Add training info
            metrics['epochs_trained'] = epoch + 1
            metrics['best_val_acc'] = best_val_acc

        return metrics

    def objective_optuna(self, trial: optuna.Trial, model_type: str) -> float:
        """
        Optuna objective function for hyperparameter optimization

        Args:
            trial: Optuna trial object
            model_type: Type of model to optimize

        Returns:
            Objective value (higher is better)
        """

        # Sample common hyperparameters
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.2, 0.7, step=0.1),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
            'epochs': 300  # Fixed with early stopping
        }

        # Add model-specific parameters
        if 'msfl' in model_type:
            # For MSFL, ensure lambdas are reasonable
            lambda_node = trial.suggest_float('lambda_node', 0.0, 0.5)
            lambda_neighbor = trial.suggest_float('lambda_neighbor', 0.0, 0.5)
            lambda_graph = trial.suggest_float('lambda_graph', 0.0, 0.5)

            # Normalize to sum to 1 (excluding accuracy)
            total = lambda_node + lambda_neighbor + lambda_graph
            if total > 0:
                params['lambda_node'] = lambda_node / total
                params['lambda_neighbor'] = lambda_neighbor / total
                params['lambda_graph'] = lambda_graph / total
            else:
                params['lambda_node'] = 0.33
                params['lambda_neighbor'] = 0.33
                params['lambda_graph'] = 0.34

            params['lambda_accuracy'] = trial.suggest_float('lambda_accuracy', 0.5, 1.5)
            params['similarity_threshold'] = trial.suggest_float('similarity_threshold', 0.6, 0.9)
            params['neighborhood_hops'] = trial.suggest_int('neighborhood_hops', 1, 3)

        
        # Create model and loss
        try:
            model, loss_fn = self.create_model_and_loss(model_type, params)

            # Train and evaluate
            metrics = self.train_and_evaluate(model, loss_fn, params)

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
                    'auc_roc': metrics['performance'].get('auc_roc', 0),
                    'f1_macro': metrics['performance'].get('f1_macro', 0),
                    'precision_per_class': metrics['performance'].get('precision_per_class', []),
                    'recall_per_class': metrics['performance'].get('recall_per_class', []),
                    'f1_per_class': metrics['performance'].get('f1_per_class', []),
                    'spd': spd,
                    'eod': eod
                }
            }
            self.results['trials'].append(trial_result)

            # Save intermediate results after each trial
            self.save_trial_results(trial.number, model_type, trial_result)

            # Print detailed metrics
            print(f"\nTrial {trial.number} Results:")
            print(f"  Accuracy: {accuracy:.4f} | Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  AUC-ROC: {metrics['performance'].get('auc_roc', 0):.4f}")
            print(f"  F1 Score (macro): {metrics['performance'].get('f1_macro', 0):.4f}")

            # Per-class metrics
            prec_per_class = metrics['performance'].get('precision_per_class', [])
            rec_per_class = metrics['performance'].get('recall_per_class', [])
            f1_per_class = metrics['performance'].get('f1_per_class', [])

            for i in range(len(prec_per_class)):
                print(f"  Class {i}: Prec={prec_per_class[i]:.3f}, Rec={rec_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")

            print(f"  Statistical Parity Diff: {spd:.4f}")
            print(f"  Equal Opportunity Diff: {eod:.4f}")
            print(f"  Objective Score: {objective:.4f}")

            # Warning if model is collapsing
            if 'warning_prediction_collapse' in metrics['performance']:
                print(f"WARNING: {metrics['performance']['warning_prediction_collapse']}")

            return objective

        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            return 0.0

    def optuna_search(self, model_type: str, n_trials: int = 50, timeout: int = None) -> optuna.Study:
        """
        Perform Optuna-based hyperparameter optimization

        Args:
            model_type: Type of model to optimize
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (optional)

        Returns:
            Optuna study object
        """

        print(f"Starting Optuna optimization for {model_type}")
        print(f"Number of trials: {n_trials}")

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
            n_jobs=1  # Set to -1 for parallel execution
        )

        # Print results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial:")
        print(f"  Value: {study.best_value:.4f}")
        print(f"  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

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

        print(f"  Trial results saved to {filename}")

    def save_intermediate_results(self, model_type: str, study, metrics: dict):
        """Save intermediate results after each model completes"""

        # Create results directory if it doesn't exist
        os.makedirs('results/intermediate', exist_ok=True)

        # Save this model's results
        filename = f"results/intermediate/{self.dataset_name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        model_results = {
            'dataset': self.dataset_name,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'best_trial': {
                'value': study.best_value,
                'params': study.best_params,
                'metrics': metrics
            },
            'all_trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': str(t.state)
                }
                for t in study.trials
            ]
        }

        with open(filename, 'w') as f:
            json.dump(model_results, f, indent=2, default=str)

        print(f"Intermediate results saved to {filename}")

    def save_results(self, comparison_df: pd.DataFrame, filename: str = None):
        """Save tuning results to files"""

        if filename is None:
            filename = f"hyperparameter_tuning_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
            f.write("HYPERPARAMETER TUNING REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write("BEST MODEL COMPARISON:\n")
            f.write("-"*40 + "\n")

            for _, row in comparison_df.iterrows():
                f.write(f"\nModel: {row['model_type']}\n")
                f.write(f"  Accuracy: {row['accuracy']:.4f}\n")
                f.write(f"  SPD: {row['spd']:.4f}\n")
                f.write(f"  EOD: {row['eod']:.4f}\n")
                f.write(f"  Objective: {row['objective']:.4f}\n")

                if isinstance(row['best_params'], dict):
                    f.write("  Best Parameters:\n")
                    for key, value in row['best_params'].items():
                        f.write(f"    {key}: {value}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("WINNER: " + comparison_df.iloc[0]['model_type'] + "\n")
            f.write("="*80 + "\n")

        print(f"Report saved to {report_path}")


def main():
    """Main function for MSFL hyperparameter tuning"""

    import argparse

    parser = argparse.ArgumentParser(description='MSFL Hyperparameter Tuning')
    parser.add_argument('--dataset', type=str, default='credit',
                       choices=['credit', 'german', 'bail'],
                       help='Dataset to use')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--mode', type=str, default='optimize',
                       choices=['optimize'],
                       help='Mode: optimize')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer trials')
    parser.add_argument('--model_type', type=str, default='msfl',
                       help='Specific model to tune (if None, tunes all)')
    args = parser.parse_args()

    # Initialize tuner
    tuner = MSFLHyperparameterTuning(
        dataset_name=args.dataset,
        data_dir='dataset'
    )

    if args.quick:
        args.n_trials = 10
    try:
        if args.mode in ['optimize']:
            # Run main optimization
            study = tuner.optuna_search(args.model_type, n_trials=args.n_trials)

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print("Partial results have been saved.")
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nMSFL hyperparameter tuning complete!")


if __name__ == "__main__":
    main()