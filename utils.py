import torch
from texttable import Texttable

def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_metrics(metrics):
    table = Texttable()
    table.add_row(["Metric", "Value"])
    for key, value in metrics.items():
        table.add_row([key, value])
    print(table.draw())
    