import torch
from jsonargparse import CLI
from model import GNN, train, test
from utils import set_device, print_metrics
from preprocess_data import preprocess_data

def main(
    data_path: str = 'dataset',
    data_name: str = 'credit',
    model: str = 'GCN',
    fame: bool = False,
    enhanced: bool = False,
    layers: int = 4,
    hidden: int = 32,
    dropout: float = 0.5,
    epochs: int = 20,
    lr: float = 0.01,
    verbose: bool = False,
):
    data, sens_attributes = preprocess_data(data_path, data_name, train_split=0.8, test_split=0.1)
    
    print(f"Training a {model} model (fame: {fame}, enhanced: {enhanced}) on {data_name} dataset with {layers} layers, {hidden} hidden units, and dropout rate of {dropout}")
    model = GNN(data, model, fame, enhanced, sens_attributes, layers=layers, hidden=hidden, dropout=dropout)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    device = set_device()
    print(f"Device: {device}")
    model.to(device)
    data.to(device)
    sens_attributes.to(device)

    model.train()
    print('\n' + "#"*25 + " Training Model " + "#"*25 + "\n")
    train(model, data, optimizer, epochs)

    model.eval()
    metrics = test(model, data, sens_attributes, verbose)

    print('\n' + "#"*25 + " Test Metrics " + "#"*25 + "\n")
    print_metrics(metrics)

if __name__=="__main__":
    CLI(main)