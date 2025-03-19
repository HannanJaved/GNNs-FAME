import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class FAME(MessagePassing):
    def __init__(self, in_channels, out_channels, sens_attribute_tensor):
        super(FAME, self).__init__(aggr='mean')  
        self.lin = nn.Linear(in_channels, out_channels)
        self.sensitive_attr = sens_attribute_tensor
        self.bias_correction = nn.Parameter(torch.rand(1))

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        group_difference = self.sensitive_attr[row] - self.sensitive_attr[col]
        
        fairness_adjustment = (1 + self.bias_correction * group_difference.view(-1, 1))

        return fairness_adjustment * norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out