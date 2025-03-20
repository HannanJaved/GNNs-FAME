from fame import FAME, A_FAME
from calculate_fairness import calculate_fairness
from preprocess_data import preprocess_data
from jsonargparse import CLI

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN(torch.nn.Module):
    def __init__(self, data, fame=False, layers=2, hidden=16, dropout=0.5):
        super(GCN, self).__init__()

        if fame:
            conv = FAME
        else:
            conv = GCNConv
        
        self.conv1 = conv(data.num_node_features, hidden)
        self.convs = torch.nn.ModuleList()
        
        for i in range(layers - 1):
            self.convs.append(conv(hidden, hidden))
        
        self.conv2 = conv(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, *args, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, data, fame=False, layers=2, hidden=16, dropout=0.5):
        super(GAT, self).__init__()
        
        if fame:
            conv = FAME
        else:
            conv = GATConv
        
        self.conv1 = conv(data.num_node_features, hidden)
        self.convs = torch.nn.ModuleList()
        
        for i in range(layers - 1):
            self.convs.append(conv(hidden, hidden))
        
        self.conv2 = conv(hidden, 2)
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
    model: str = 'GCN',
    fame: bool = False,
    layers: int = 2,
    hidden: int = 16,
    dropout: float = 0.5,
    epochs: int = 100,
):
    data = preprocess_data(data_path, data_name, train_split=0.8, test_split=0.1)
    
    if model == 'GCN':
        model = GCN(data, fame=fame, layers=layers, hidden=hidden, dropout=dropout)
    elif model == 'GAT':
        model = GAT(data, fame=fame, layers=layers, hidden=hidden, dropout=dropout)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    device = set_device()
    model.to(device)
    data.to(device)
    
    model.train()
    train(model, data, optimizer, epochs)

    model.eval()
    test(model, data)

def train(model, data, optimizer, epochs):
    for epoch in range(epochs):
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
def test(model, data):
    with torch.inference_mode():
      out = model(data.x, data.edge_index)

    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    accuracy = correct / int(data.test_mask.sum())
    
    predictions = out.argmax(dim=1)

    fairness_metrics = calculate_fairness(data, predictions, data.sensitive_attribute)
    fairness_metrics['Accuracy'] = accuracy

if __name__=="__main__":
    CLI(main)