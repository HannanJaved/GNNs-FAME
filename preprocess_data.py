import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

def preprocess_data(data_dir, data_name, train_split, test_split=0.1):
    """"
    Function to preprocess the datasets from the data directory
    
    Args:
        data_dir: str - Path to the directory containing the dataset
        data_name: str - Name of the dataset
        train_split: float - Fraction of the dataset to be used for training
        test_split: float - Fraction of the dataset to be used for testing

    Returns:
        data: torch_geometric.data.Data - Data object containing the node features, edge index, and labels
        sens_attribute_tensor: torch.Tensor - Tensor containing the sensitive attribute values
    """
    if data_name not in ["credit", "german", "pokec-z"]:
        print("Invalid dataset name. Please choose from 'credit', 'german', or 'pokec-z'")
        return
    
    data_dir = os.path.join(data_dir, data_name)

    if data_name == "pokec-z":
        user_labels_path = os.path.join(data_dir, "region_job.csv")
        user_edges_path = os.path.join(data_dir, "region_job_relationship.csv")
    else:
        user_labels_path = os.path.join(data_dir, data_name + ".csv")
        user_edges_path = os.path.join(data_dir, data_name + "_edges.csv")

    # Create dataframes to store the information from the .csv files
    user_labels = pd.read_csv(user_labels_path)
    user_edges = pd.read_csv(user_edges_path)

    user_labels.insert(0, 'user_id', user_labels.index)

    if data_name == "german":
        user_labels['Gender'] = user_labels['Gender'].replace({'Female': 1, 'Male': 0})
        user_labels['GoodCustomer'] = user_labels['GoodCustomer'].replace({1: 1, -1: 0})
        user_labels = user_labels.drop('PurposeOfLoan', axis=1)

    user_edges = user_edges[user_edges['uid1'].isin(user_labels['user_id']) & user_edges['uid2'].isin(user_labels['user_id'])]
    user_labels_train = user_labels

    if data_name == "credit":
        user_labels_train = user_labels_train.drop(columns=['NoDefaultNextMonth'])
    elif data_name == "german":
        user_labels_train = user_labels_train.drop(columns=['GoodCustomer'])
    elif data_name == "pokec-z":
        user_labels_train = user_labels_train.drop(columns=['I_am_working_in_field'])

    # Extract node features from user_labels dataframe
    node_features = user_labels_train.iloc[:, 1:]
    node_features = torch.tensor(node_features.values, dtype=torch.float)

    # Extract edges from user_edges dataframe
    edges = user_edges[['uid1', 'uid2']]
    edges['uid1'] = edges['uid1'].map(dict(zip(user_labels['user_id'], range(len(user_labels)))))
    edges['uid2'] = edges['uid2'].map(dict(zip(user_labels['user_id'], range(len(user_labels)))))

    # Convert edges dataframe to tensor
    edges_tensor = torch.tensor(edges.values, dtype=torch.long).t().contiguous()

    # Create edge_index tensor
    edge_index = edges_tensor

    if data_name == "pokec-z":
        user_labels['I_am_working_in_field'] = user_labels['I_am_working_in_field'].map({-1: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 1})

    # Create torch-geometric data
    data = Data(x=node_features, edge_index=edge_index)

    num_nodes = node_features.size(0)

    # Create masks for training, validation, and testing
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # random split 
    if train_split + test_split > 1:
        print("Train split and test split should not be greater than 1")
        return
    else:
        train_size = train_split
        test_size = test_split
        val_size = 1 - train_size - test_size
        print("Train size: ", train_size)
        print("Validation size: ", val_size)
        print("Test size: ", test_size)

    train_indices, temp_indices = train_test_split(np.arange(num_nodes), train_size=train_size, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, train_size=val_size/(val_size + test_size), random_state=42)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Labels from the data
    if data_name == "credit":
        data.y = torch.tensor(user_labels['NoDefaultNextMonth'].values, dtype=torch.long)
        sens_attribute_tensor = torch.tensor(user_labels['Age'].values, dtype=torch.long)
    elif data_name == "german":
        data.y = torch.tensor(user_labels['GoodCustomer'].values, dtype=torch.long)
        sens_attribute_tensor = torch.tensor(user_labels['Gender'].values, dtype=torch.long)
    elif data_name == "pokec-z":
        data.y = torch.tensor(user_labels['I_am_working_in_field'].values, dtype=torch.long)
        sens_attribute_tensor = torch.tensor(user_labels['region'].values, dtype=torch.long)

    return data, sens_attribute_tensor