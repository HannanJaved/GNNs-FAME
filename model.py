import torch
from tqdm import tqdm
from fame import FAME, A_FAME
from enhanced_fame import EnhancedFAME, EnhancedAFAME
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from calculate_fairness import calculate_fairness
from torch_geometric.data import Data as torch_geometric_Data

class GNN(torch.nn.Module):
    def __init__(
        self, 
        data: torch_geometric_Data, 
        model: str = "GCN", 
        fame: bool = False, 
        enhanced: bool = False,
        sens_attribute: torch.Tensor = None, 
        layers: int = 2, 
        hidden: int = 16, 
        dropout: float = 0.5,
    ):
        super(GNN, self).__init__()

        if model == "GCN":    
            self.convs = torch.nn.ModuleList()

            if fame:
                if enhanced:
                    self.conv1 = EnhancedFAME(data.num_node_features, hidden, sens_attribute)
                    for i in range(layers - 1):
                        self.convs.append(EnhancedFAME(hidden, hidden, sens_attribute))
                    self.conv2 = EnhancedFAME(hidden, 2, sens_attribute)
                else:
                    self.conv1 = FAME(data.num_node_features, hidden, sens_attribute)
                    for i in range(layers - 1):
                        self.convs.append(FAME(hidden, hidden, sens_attribute))
                    self.conv2 = FAME(hidden, 2, sens_attribute)
            else:
                self.conv1 = GCNConv(data.num_node_features, hidden)
                for i in range(layers - 1):
                    self.convs.append(GCNConv(hidden, hidden))
                self.conv2 = GCNConv(hidden, 2)

        elif model == "GAT":    
            self.convs = torch.nn.ModuleList()

            if fame:
                if enhanced:
                    self.conv1 = EnhancedAFAME(data.num_node_features, hidden, sens_attribute)
                    for i in range(layers - 1):
                        self.convs.append(EnhancedAFAME(hidden, hidden, sens_attribute))
                    self.conv2 = EnhancedAFAME(hidden, 2, sens_attribute)
                else:
                    self.conv1 = A_FAME(data.num_node_features, hidden, sens_attribute)
                    for i in range(layers - 1):
                        self.convs.append(A_FAME(hidden, hidden, sens_attribute))
                    self.conv2 = A_FAME(hidden, 2, sens_attribute)
            else:
                self.conv1 = GATConv(data.num_node_features, hidden)
                for i in range(layers - 1):
                    self.convs.append(GATConv(hidden, hidden))
                self.conv2 = GATConv(hidden, 2)

        self.dropout = dropout

    def forward(self, x, edge_index, *args, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    

def train(
    model: torch.nn.Module, 
    data: torch_geometric_Data, 
    optimizer: torch.optim.Optimizer, 
    epochs: int,
    loss_fn: torch.nn.Module = torch.nn.NLLLoss(),
):
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                val_out = model(data.x, data.edge_index)
                val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask])
                print(f'Epoch {epoch} | Loss: {loss.item()} | Validation Loss: {val_loss.item()}')
            model.train()


@torch.no_grad()
def test(
    model: torch.nn.Module, 
    data: torch_geometric_Data, 
    sens_attributes: torch.Tensor,
    verbose: bool = False,
):
    with torch.inference_mode():
      out = model(data.x, data.edge_index)

    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    accuracy = correct / int(data.test_mask.sum())
    
    predictions = out.argmax(dim=1)
    predictions = predictions.to('cpu') 

    fairness_metrics = calculate_fairness(data, predictions, sens_attributes, verbose)
    fairness_metrics['Accuracy'] = accuracy

    return fairness_metrics