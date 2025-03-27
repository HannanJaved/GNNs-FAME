import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from utils import set_device
from calculate_fairness import calculate_fairness

class EnhancedFAME(MessagePassing):
    def __init__(self, in_channels, out_channels, sens_attribute_tensor, fairness_weight=1.0, adaptive=True):
        super(EnhancedFAME, self).__init__(aggr='mean')  
        self.device = set_device()
        self.lin = Linear(in_channels, out_channels).to(self.device)
        self.sensitive_attr = sens_attribute_tensor.clone().to(self.device)
        
        # Enhanced fairness parameters
        self.bias_correction = Parameter(torch.rand(1, device=self.device))
        self.fairness_weight = fairness_weight
        self.adaptive = adaptive
        
        # Group representation balancing
        self.group_norm = Parameter(torch.ones(2, 1, device=self.device))
        
        # Adaptive fairness metrics
        self.fairness_metrics = None
        self.adjustment_rate = 0.01

    def forward(self, x, edge_index, fairness_metrics=None):
        # Move inputs to the same device
        x_device = x.clone().to(self.device)
        edge_index_device = edge_index.clone().to(self.device)
        
        # Update fairness metrics if adaptive mode is enabled
        if self.adaptive and fairness_metrics is not None:
            self.fairness_metrics = fairness_metrics
            self._adjust_correction()
            
        edge_index_device, _ = add_self_loops(edge_index_device, num_nodes=x_device.size(0))
        
        # Apply linear transformation only to the feature matrix x
        x_device = self.lin(x_device)
        
        # Pass only the necessary data to propagate
        return self.propagate(edge_index_device, size=(x_device.size(0), x_device.size(0)), x=x_device)
    
    def _adjust_correction(self):
        """Adaptively adjust bias correction based on current fairness metrics"""
        if self.fairness_metrics is not None:
            # Use statistical parity difference as a guide
            spd = self.fairness_metrics.get('Statistical Parity Difference', 0)
            # Increase correction if unfairness is high
            adjustment = self.adjustment_rate * spd
            with torch.no_grad():
                self.bias_correction.add_(adjustment)
    
    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_device = deg.clone().to(self.device)  # Clone before moving to device
        deg_inv_sqrt = deg_device.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Enhanced group difference handling
        group_difference = self.sensitive_attr[row] - self.sensitive_attr[col]
        
        # Apply group-specific normalization - Fix the dtype mismatch
        group_norm_factors = torch.ones_like(group_difference, dtype=torch.float, device=self.device)
        # Convert boolean masks to indices for proper assignment
        idx_group0 = (self.sensitive_attr[row] == 0).nonzero(as_tuple=True)[0]
        idx_group1 = (self.sensitive_attr[row] == 1).nonzero(as_tuple=True)[0]
        
        if len(idx_group0) > 0:
            group_norm_factors[idx_group0] = self.group_norm[0]
        if len(idx_group1) > 0:
            group_norm_factors[idx_group1] = self.group_norm[1]
        
        # Enhanced fairness adjustment with weight control
        group_difference = group_difference.to(self.device)
        fairness_adjustment = (1 + self.fairness_weight * self.bias_correction * 
                               group_difference.view(-1, 1))
        
        # Apply group normalization to message
        return fairness_adjustment * norm.view(-1, 1) * x_j * group_norm_factors.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out
    
    def fairness_regularization_loss(self):
        """Regularization loss to prevent extreme fairness corrections"""
        return torch.abs(self.bias_correction).mean()


class EnhancedAFAME(MessagePassing):
    def __init__(self, in_channels, out_channels, sens_attribute_tensor, fairness_weight=1.0, adaptive=True):
        super(EnhancedAFAME, self).__init__(aggr='add') 
        self.device = set_device()
        self.lin = Linear(in_channels, out_channels).to(self.device)
        self.att = Linear(2 * out_channels, 1).to(self.device)
        self.sensitive_attr = sens_attribute_tensor.clone().to(self.device)
        
        # Enhanced fairness parameters
        self.bias_correction = Parameter(torch.rand(1, device=self.device))
        self.fairness_weight = fairness_weight
        self.adaptive = adaptive
        
        # Group-specific attention weights
        self.group_attention = Parameter(torch.ones(2, device=self.device))
        
        # Adaptive fairness metrics
        self.fairness_metrics = None
        self.adjustment_rate = 0.01
        
        # Edge fairness memory for persistent corrections
        self.edge_fairness_memory = None

    def forward(self, x, edge_index, fairness_metrics=None):
        # Move inputs to the same device
        x_device = x.clone().to(self.device)
        edge_index_device = edge_index.clone().to(self.device)
        
        # Update fairness metrics if adaptive mode is enabled
        if self.adaptive and fairness_metrics is not None:
            self.fairness_metrics = fairness_metrics
            self._adjust_correction()
            
        edge_index_device, _ = add_self_loops(edge_index_device, num_nodes=x_device.size(0))
        
        # Initialize edge fairness memory if not exists
        if self.edge_fairness_memory is None or self.edge_fairness_memory.size(0) != edge_index_device.size(1):
            self.edge_fairness_memory = torch.zeros(edge_index_device.size(1), 1, device=self.device)
        
        x_device = self.lin(x_device)
        
        return self.propagate(edge_index_device, size=(x_device.size(0), x_device.size(0)), x=x_device)
    
    def _adjust_correction(self):
        """Adaptively adjust bias correction based on current fairness metrics"""
        if self.fairness_metrics is not None:
            # Use equal opportunity difference as a guide for attention mechanism
            eod = self.fairness_metrics.get('Equal Opportunity Difference', 0)
            # Increase correction if unfairness is high
            adjustment = self.adjustment_rate * eod
            with torch.no_grad():
                self.bias_correction.add_(adjustment)
                
                # Also adjust group attention weights based on overall accuracy
                acc_g0 = self.fairness_metrics.get('Overall Accuracy Group with S=0', 0.5)
                acc_g1 = self.fairness_metrics.get('Overall Accuracy Group S=1', 0.5)
                if acc_g0 < acc_g1:
                    self.group_attention[0].add_(self.adjustment_rate)
                else:
                    self.group_attention[1].add_(self.adjustment_rate)

    def message(self, edge_index, x_i, x_j, size_i):
        x_cat = torch.cat([x_i, x_j], dim=-1)  
        alpha = self.att(x_cat)

        row, col = edge_index
        group_difference = self.sensitive_attr[row] - self.sensitive_attr[col]

        # Enhanced fairness adjustment with persistence
        group_difference = group_difference.to(self.device)
        fairness_adjustment = self.fairness_weight * self.bias_correction * group_difference.view(-1, 1)
        
        # Update edge fairness memory with exponential moving average
        self.edge_fairness_memory = 0.9 * self.edge_fairness_memory + 0.1 * fairness_adjustment
        
        # Apply fairness adjustment to attention scores
        alpha = alpha + self.edge_fairness_memory
        
        # Group-specific attention scaling - Fix the dtype mismatch
        group_weights = torch.ones_like(group_difference, dtype=torch.float, device=self.device)
        # Convert boolean masks to indices for proper assignment
        idx_group0 = (self.sensitive_attr[row] == 0).nonzero(as_tuple=True)[0]
        idx_group1 = (self.sensitive_attr[row] == 1).nonzero(as_tuple=True)[0]
        
        if len(idx_group0) > 0:
            group_weights[idx_group0] = self.group_attention[0]
        if len(idx_group1) > 0:
            group_weights[idx_group1] = self.group_attention[1]
        
        alpha = alpha * group_weights.view(-1, 1)
        
        # Apply softmax normalization
        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)

        return alpha * x_j

    def update(self, aggr_out):
        return aggr_out
    
    def fairness_regularization_loss(self):
        """Regularization loss to prevent extreme fairness corrections"""
        group_balance_loss = torch.abs(self.group_attention[0] - self.group_attention[1])
        correction_loss = torch.abs(self.bias_correction).mean()
        return correction_loss + 0.1 * group_balance_loss


# Trainer function to use the enhanced models with fairness regularization
def train_with_fairness(
    model, 
    data, 
    optimizer, 
    epochs, 
    sens_attributes,
    fairness_reg_weight=0.1
):
    """Training function that incorporates fairness regularization"""
    device = next(model.parameters()).device
    
    # Don't modify the original data, just create device copies as needed
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Clone and move inputs to the device for this forward pass
        x_device = data.x.clone().to(device)
        edge_index_device = data.edge_index.clone().to(device)
        out = model(x_device, edge_index_device)
        
        criterion = torch.nn.NLLLoss()
        # Create device copies of tensors needed for loss calculation
        y_device = data.y.clone().to(device)
        train_mask_device = data.train_mask.clone().to(device)
        
        # Use device-appropriate tensors for loss calculation
        loss = criterion(out[train_mask_device], y_device[train_mask_device])
        
        # Add fairness regularization if model supports it
        fairness_reg_loss = 0
        if hasattr(model, 'conv1') and hasattr(model.conv1, 'fairness_regularization_loss'):
            fairness_reg_loss += model.conv1.fairness_regularization_loss()
            
        if hasattr(model, 'conv2') and hasattr(model.conv2, 'fairness_regularization_loss'):
            fairness_reg_loss += model.conv2.fairness_regularization_loss()
            
        if hasattr(model, 'convs'):
            for conv in model.convs:
                if hasattr(conv, 'fairness_regularization_loss'):
                    fairness_reg_loss += conv.fairness_regularization_loss()
        
        # Combine losses with regularization weight
        total_loss = loss + fairness_reg_weight * fairness_reg_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Every 10 epochs, evaluate current fairness and update adaptive parameters
        if epoch % 10 == 0 and epoch > 0:
            with torch.no_grad():
                predictions = out.argmax(dim=1)
                fairness_metrics = calculate_fairness(data, predictions, sens_attributes)
                
                # Update fairness parameters in the model if it supports adaptive fairness
                if hasattr(model, 'conv1'):
                    if hasattr(model.conv1, 'adaptive') and model.conv1.adaptive:
                        # Use the forward method correctly - don't process data.x again if not needed
                        model.conv1._adjust_correction() if hasattr(model.conv1, '_adjust_correction') else None
                        model.conv1.fairness_metrics = fairness_metrics
                
                if hasattr(model, 'conv2'):
                    if hasattr(model.conv2, 'adaptive') and model.conv2.adaptive:
                        # Just update the fairness metrics and adjustment without recomputing forward pass
                        model.conv2._adjust_correction() if hasattr(model.conv2, '_adjust_correction') else None
                        model.conv2.fairness_metrics = fairness_metrics
                
                if hasattr(model, 'convs'):
                    for conv in model.convs:
                        if hasattr(conv, 'adaptive') and conv.adaptive:
                            # Update fairness metrics without full forward pass
                            conv._adjust_correction() if hasattr(conv, '_adjust_correction') else None
                            conv.fairness_metrics = fairness_metrics
                
                print(f'Epoch {epoch} | Loss: {loss.item()} | Fairness Loss: {fairness_reg_loss.item()}')
                print(f'SPD: {fairness_metrics["Statistical Parity Difference"]:.4f} | EOD: {fairness_metrics["Equal Opportunity Difference"]:.4f}')