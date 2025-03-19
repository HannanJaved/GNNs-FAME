import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from layers.fame import FAME
from layers.a_fame import A_FAME
from jsonargparse import CLI

class Fair_GNN(torch.nn.Module):
    def __init__(self, data, layers=1, hidden=128, dropout=0):
        super(Fair_GNN, self).__init__()
        # sens_attribute_tensor = torch.tensor(data.sensitive_attr.values, dtype=torch.long)
        self.conv1 = FAME(data.num_node_features, hidden, sens_attribute_tensor)
        # self.conv1 = A_FAME(data.num_node_features, hidden, sens_attribute_tensor)
        self.fc = Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, *args, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def load_data():
    node_data = pd.read_csv('dataset/bail/bail.csv')
    edge_data = np.loadtxt('dataset/bail/bail_edges.txt', dtype=int)
    
    x = torch.tensor(node_data.values, dtype=torch.float)
    edge_index = torch.tensor(edge_data.T, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    return data

def main(
    data,
    layers: int = 1,
    hidden: int = 128,
    dropout: float = 0.5,
    seed: int = 42,
    **kwargs
):
    torch.manual_seed(seed)
    data = load_data()
    model = Fair_GNN(data, layers, hidden, dropout)
    return model

if __name__=="__main__":
    CLI(main)