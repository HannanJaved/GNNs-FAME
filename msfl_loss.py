import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, degree, to_dense_adj
from torch_scatter import scatter_mean
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import global_mean_pool


class MultiScaleFairnessLoss(nn.Module):
    """
    Multi-Scale Fairness Loss combining three levels of fairness.
    """

    def __init__(self,
                 lambda_node=0.3,      # Weight for individual fairness
                 lambda_neighbor=0.3,  # Weight for neighborhood fairness
                 lambda_graph=0.4,     # Weight for global fairness
                 lambda_accuracy=1.0,  # Weight for task accuracy
                 similarity_threshold=0.8,  # Threshold for "similar" nodes
                 neighborhood_hops=2,       # Hops for neighborhood definition
                 class_weights=None):       # Class weights for handling imbalanced datasets

        super(MultiScaleFairnessLoss, self).__init__()
        self.lambda_node = lambda_node
        self.lambda_neighbor = lambda_neighbor
        self.lambda_graph = lambda_graph
        self.lambda_accuracy = lambda_accuracy
        self.similarity_threshold = similarity_threshold
        self.neighborhood_hops = neighborhood_hops
        self.class_weights = class_weights

    def compute_node_level_fairness(self, embeddings, predictions, sensitive_attr):
        """
        Node-level (Individual) Fairness: Similar individuals should receive similar outcomes

        """
        n_nodes = embeddings.size(0)
        device = embeddings.device
        batch_size = 100  # Process nodes in batches to save memory

        # Normalize embeddings once
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # Convert predictions to probabilities
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # Multi-class: use softmax
            pred_probs = torch.softmax(predictions, dim=1)[:, 1] if predictions.size(1) == 2 else predictions.max(dim=1)[0]
        else:
            # Binary: use sigmoid
            pred_probs = torch.sigmoid(predictions.squeeze())

        # Accumulate unfairness across batches
        total_unfairness = 0.0
        total_valid_pairs = 0.0

        # Process nodes in batches to save memory
        for batch_start in range(0, n_nodes, batch_size):
            batch_end = min(batch_start + batch_size, n_nodes)
            batch_indices = torch.arange(batch_start, batch_end, device=device)

            # Compute similarity for this batch with all nodes
            batch_emb = embeddings_norm[batch_indices]
            similarity_batch = torch.mm(batch_emb, embeddings_norm.t())
            similar_pairs = (similarity_batch > self.similarity_threshold).float()

            # Check if nodes have different sensitive attributes
            batch_sensitive = sensitive_attr[batch_indices]
            s_diff = (batch_sensitive.unsqueeze(1) != sensitive_attr.unsqueeze(0)).float()

            # Prediction difference (using probabilities)
            batch_pred = pred_probs[batch_indices]
            pred_diff = torch.abs(batch_pred.unsqueeze(1) - pred_probs.unsqueeze(0))

            # Accumulate unfairness
            batch_unfairness = (similar_pairs * s_diff * pred_diff).sum()
            batch_valid_pairs = (similar_pairs * s_diff).sum()

            total_unfairness += batch_unfairness
            total_valid_pairs += batch_valid_pairs

        # Individual fairness: Similar nodes with different sensitive attrs should have similar predictions
        individual_unfairness = total_unfairness / total_valid_pairs.clamp(min=1)

        return individual_unfairness

    def compute_neighborhood_level_fairness(self, predictions, sensitive_attr, edge_index, node_features=None):
        """
        Neighborhood-level Fairness: Within local communities, groups should have similar outcomes

        """
        device = predictions.device
        n_nodes = predictions.size(0)

        # Convert predictions to probabilities
        if predictions.dim() > 1 and predictions.size(1) > 1:
            pred_probs = torch.softmax(predictions, dim=1)[:, 1] if predictions.size(1) == 2 else predictions.max(dim=1)[0]
        else:
            pred_probs = torch.sigmoid(predictions.squeeze())

        # Find neighborhoods using k-hop subgraphs
        neighborhood_fairness = 0.0
        num_neighborhoods = 0

        # Sample nodes to check their neighborhoods (can sample for efficiency)
        sample_size = min(100, n_nodes)  # Check 100 random neighborhoods
        sampled_nodes = torch.randperm(n_nodes, device=device)[:sample_size]

        for center_node in sampled_nodes:
            # Get k-hop neighborhood
            # Convert center_node to tensor for k_hop_subgraph
            center_node_tensor = torch.tensor([center_node.item()], dtype=torch.long, device=edge_index.device)
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                center_node_tensor,
                self.neighborhood_hops,
                edge_index,
                relabel_nodes=True,
                num_nodes=n_nodes  # Use full number of nodes in the graph
            )

            if len(subset) < 5:  # Skip tiny neighborhoods
                continue

            # Get predictions and sensitive attributes for this neighborhood
            neigh_preds = pred_probs[subset]
            neigh_sensitive = sensitive_attr[subset]

            # Calculate fairness within this neighborhood
            if (neigh_sensitive == 0).any() and (neigh_sensitive == 1).any():
                # Mean prediction for each group in this neighborhood
                pred_s0 = neigh_preds[neigh_sensitive == 0].mean()
                pred_s1 = neigh_preds[neigh_sensitive == 1].mean()

                # Fairness violation in this neighborhood
                neighborhood_fairness += torch.abs(pred_s0 - pred_s1)
                num_neighborhoods += 1

        if num_neighborhoods > 0:
            neighborhood_fairness /= num_neighborhoods

        return neighborhood_fairness

    def compute_graph_level_fairness(self, predictions, sensitive_attr):
        """
        Graph-level (Global) Fairness: Statistical parity across entire graph

        """
        # Statistical Parity Difference
        if predictions.dim() > 1 and predictions.size(1) > 1:
            pred_probs = torch.softmax(predictions, dim=1)[:, 1] if predictions.size(1) == 2 else predictions.max(dim=1)[0]
        else:
            pred_probs = torch.sigmoid(predictions.squeeze())

        if (sensitive_attr == 0).any() and (sensitive_attr == 1).any():
            prob_s0 = pred_probs[sensitive_attr == 0].mean()
            prob_s1 = pred_probs[sensitive_attr == 1].mean()
            spd = torch.abs(prob_s0 - prob_s1)
        else:
            spd = torch.tensor(0.0, device=predictions.device)

        return spd

    def forward(self, predictions, labels, sensitive_attr, embeddings, edge_index, mask=None):
        """
        Combine all three scales of fairness with task loss

        """
        # If mask is provided, compute loss only on masked nodes
        if mask is not None:
            masked_predictions = predictions[mask]
            masked_labels = labels[mask]
            masked_sensitive = sensitive_attr[mask]
            masked_embeddings = embeddings[mask]
        else:
            masked_predictions = predictions
            masked_labels = labels
            masked_sensitive = sensitive_attr
            masked_embeddings = embeddings

        # Task accuracy loss (on masked nodes only)
        if masked_predictions.dim() > 1 and masked_predictions.size(1) > 1:
            # Use class weights if provided for handling class imbalance
            accuracy_loss = F.cross_entropy(masked_predictions, masked_labels.long(), weight=self.class_weights)
        else:
            # For binary classification with single output
            if self.class_weights is not None:
                # Apply class weights manually for BCEWithLogitsLoss
                pos_weight = self.class_weights[1] / self.class_weights[0] if self.class_weights[0] != 0 else torch.tensor(1.0)
                accuracy_loss = F.binary_cross_entropy_with_logits(
                    masked_predictions.squeeze(),
                    masked_labels.float(),
                    pos_weight=pos_weight
                )
            else:
                accuracy_loss = F.binary_cross_entropy_with_logits(masked_predictions.squeeze(), masked_labels.float())

        # Three scales of fairness (computed on all nodes for neighborhood structure)
        node_fairness = self.compute_node_level_fairness(embeddings, predictions, sensitive_attr)
        neighbor_fairness = self.compute_neighborhood_level_fairness(predictions, sensitive_attr, edge_index)
        graph_fairness = self.compute_graph_level_fairness(predictions, sensitive_attr)

        # Combined multi-scale loss 
        total_loss = (self.lambda_accuracy * accuracy_loss +
                     self.lambda_node * node_fairness +
                     self.lambda_neighbor * neighbor_fairness +
                     self.lambda_graph * graph_fairness)

        # Return detailed loss components for analysis
        return {
            'total_loss': total_loss,
            'accuracy_loss': accuracy_loss,
            'node_fairness_loss': node_fairness,
            'neighbor_fairness_loss': neighbor_fairness, 
            'graph_fairness_loss': graph_fairness,
            'fairness_total': (self.lambda_node * node_fairness +
                             self.lambda_neighbor * neighbor_fairness +
                             self.lambda_graph * graph_fairness)
        }

