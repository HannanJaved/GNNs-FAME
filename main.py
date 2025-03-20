from fame import FAME, A_FAME
from calculate_fairness import calculate_fairness
from preprocess_data import preprocess_data
from jsonargparse import CLI

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from texttable import Texttable
from tqdm import tqdm

def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_metrics(metrics):
    table = Texttable()
    table.add_row(["Metric", "Value"])
    for key, value in metrics.items():
        table.add_row([key, value])
    print(table.draw())
    
class GNN(torch.nn.Module):
    def __init__(
        self, 
        data: torch_geometric.data.Data, 
        model: str = "GCN", 
        fame: bool = False, 
        sens_attribute: torch.Tensor = None, 
        layers: int = 2, 
        hidden: int = 16, 
        dropout: float = 0.5,
    ):
        super(GNN, self).__init__()

        if model == "GCN":    
            self.convs = torch.nn.ModuleList()

            if fame:
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
    

def main(
    data_path: str = 'dataset',
    data_name: str = 'german',
    model: str = 'GAT',
    fame: bool = True,
    layers: int = 2,
    hidden: int = 16,
    dropout: float = 0.5,
    epochs: int = 100,
):
    data, sens_attributes = preprocess_data(data_path, data_name, train_split=0.8, test_split=0.1)
    
    print(f"Training a {model} model (fame: {fame}) on {data_name} dataset with {layers} layers, {hidden} hidden units, and dropout rate of {dropout}")
    model = GNN(data, model, fame, sens_attributes, layers=layers, hidden=hidden, dropout=dropout)    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    device = set_device()
    print(f"Device: {device}")
    model.to(device)
    data.to(device)
    sens_attributes.to(device)

    model.train()
    print('\n' + "#"*25 + " Training Model " + "#"*25 + "\n")
    train(model, data, optimizer, epochs)

    model.eval()
    metrics = test(model, data, sens_attributes)

    print('\n' + "#"*25 + " Test Metrics " + "#"*25 + "\n")
    print_metrics(metrics)

def train(
    model: torch.nn.Module, 
    data: torch_geometric.data.Data, 
    optimizer: torch.optim.Optimizer, 
    epochs: int,
):
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        criterion = torch.nn.NLLLoss()
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
                print(f'Epoch {epoch} | Loss: {loss.item()} | Validation Loss: {val_loss.item()}')
            model.train()


@torch.no_grad()
def test(
    model: torch.nn.Module, 
    data: torch_geometric.data.Data, 
    sens_attributes: torch.Tensor,
):
    with torch.inference_mode():
      out = model(data.x, data.edge_index)

    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    accuracy = correct / int(data.test_mask.sum())
    
    predictions = out.argmax(dim=1)
    predictions = predictions.to('cpu') 

    fairness_metrics = calculate_fairness(data, predictions, sens_attributes)
    fairness_metrics['Accuracy'] = accuracy

    return fairness_metrics

if __name__=="__main__":
    CLI(main)