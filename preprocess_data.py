import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def _create_masks(num_nodes, train_split, test_split, seed=42):
    """Helper function to create train, validation, and test masks."""
    if train_split + test_split > 1.0:
        raise ValueError("Sum of train_split and test_split cannot exceed 1.")
    
    val_split = 1.0 - train_split - test_split
    
    indices = np.arange(num_nodes)
    
    train_indices, temp_indices = train_test_split(
        indices, train_size=train_split, random_state=seed, stratify=None
    )
    
    # Adjust val_size to be a fraction of the remaining data
    if val_split + test_split > 0:
        relative_val_size = val_split / (val_split + test_split)
        val_indices, test_indices = train_test_split(
            temp_indices, train_size=relative_val_size, random_state=seed, stratify=None
        )
    else: # Handle case where train_split is 1.0
        val_indices, test_indices = np.array([]), np.array([])

    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    print(f"Data split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")
    return train_mask, val_mask, test_mask

def _remap_edges(user_labels, user_edges):
    """Helper function to remap edge indices based on filtered user labels."""
    user_labels = user_labels.reset_index(drop=True)
    
    user_edges = user_edges[
        user_edges['uid1'].isin(user_labels['user_id']) & 
        user_edges['uid2'].isin(user_labels['user_id'])
    ].copy()

    id_to_idx_map = {user_id: i for i, user_id in enumerate(user_labels['user_id'])}
    
    user_edges['uid1'] = user_edges['uid1'].map(id_to_idx_map)
    user_edges['uid2'] = user_edges['uid2'].map(id_to_idx_map)
    
    edge_index = torch.tensor(user_edges[['uid1', 'uid2']].values, dtype=torch.long).t().contiguous()
    return edge_index

def _load_credit_data(data_dir):
    """Loads the Credit Defaulter dataset."""
    df_path = os.path.join(data_dir, "credit.csv")
    edges_path = os.path.join(data_dir, "credit_edges.csv")
    
    user_labels = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    user_labels.insert(0, 'user_id', user_labels.index)

    # Sensitive Attribute: Age
    sens_attr = torch.tensor(user_labels['Age'].values, dtype=torch.long)
    # Task: Predict whether a user will default
    labels = torch.tensor(user_labels['NoDefaultNextMonth'].values, dtype=torch.long)
    
    # Features: All columns except user_id, label, and sensitive attribute
    node_features_df = user_labels.drop(columns=['user_id', 'NoDefaultNextMonth', 'Age'])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)

    edge_index = _remap_edges(user_labels, user_edges)
    
    return Data(x=node_features, edge_index=edge_index, y=labels), sens_attr

def _load_german_data(data_dir):
    """Loads the German Credit dataset."""
    df_path = os.path.join(data_dir, "german.csv")
    edges_path = os.path.join(data_dir, "german_edges.csv")

    user_labels = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    user_labels.insert(0, 'user_id', user_labels.index)

    user_labels['Gender'] = user_labels['Gender'].replace({'Female': 1, 'Male': 0})
    user_labels['GoodCustomer'] = user_labels['GoodCustomer'].replace({1: 1, -1: 0})
    
    # Sensitive Attribute: Gender
    sens_attr = torch.tensor(user_labels['Gender'].values, dtype=torch.long)
    # Task: Classify credit risk
    labels = torch.tensor(user_labels['GoodCustomer'].values, dtype=torch.long)
    
    # Features: All columns except user_id, label, sensitive attr, and a leaky feature
    node_features_df = user_labels.drop(columns=['user_id', 'GoodCustomer', 'Gender', 'PurposeOfLoan'])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    
    edge_index = _remap_edges(user_labels, user_edges)

    return Data(x=node_features, edge_index=edge_index, y=labels), sens_attr

def _load_bail_data(data_dir):
    """Loads the Recidivism (Bail) dataset."""
    df_path = os.path.join(data_dir, "bail.csv")
    edges_path = os.path.join(data_dir, "bail_edges.csv")

    user_labels = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    user_labels.insert(0, 'user_id', user_labels.index)

    # Sensitive Attribute: Race (WHITE=1 vs. non-WHITE=0)
    sens_attr = torch.tensor(user_labels['WHITE'].values, dtype=torch.long)
    # Task: Predict recidivism
    labels = torch.tensor(user_labels['RECID'].values, dtype=torch.long)
    
    # Features: All columns except user_id, label, and sensitive attribute
    node_features_df = user_labels.drop(columns=['user_id', 'RECID', 'WHITE'])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    
    edge_index = _remap_edges(user_labels, user_edges)

    return Data(x=node_features, edge_index=edge_index, y=labels), sens_attr

def _load_nba_data(data_dir):
    """Loads the NBA player dataset."""
    df_path = os.path.join(data_dir, "nba.csv")
    edges_path = os.path.join(data_dir, "nba_edges.csv")

    user_labels = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    user_labels.insert(0, 'user_id', user_labels.index)
    
    user_labels['SALARY'] = (user_labels['SALARY'] > user_labels['SALARY'].median()).astype(int)
    user_labels['country'] = (user_labels['country'] == 'USA').astype(int)

    sens_attr = torch.tensor(user_labels['country'].values, dtype=torch.long)
    labels = torch.tensor(user_labels['SALARY'].values, dtype=torch.long)
    
    node_features_df = user_labels.drop(columns=['user_id', 'Player', 'SALARY', 'country', 'Draft_Yr', 'TEAM'])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    
    edge_index = _remap_edges(user_labels, user_edges)

    return Data(x=node_features, edge_index=edge_index, y=labels), sens_attr

def _load_pokec_data(data_dir, region_name):
    """Loads a region-specific Pokec dataset (pokec-z or pokec-n)."""
    df_path = os.path.join(data_dir, "region_job.csv")
    edges_path = os.path.join(data_dir, "region_job_relationship.csv")

    user_labels_full = pd.read_csv(df_path)
    user_edges = pd.read_csv(edges_path)
    
    region_map = {
        'pokec-z': 'zilinasky kraj, zilina',
        'pokec-n': 'nitriansky kraj, nitra'
    }
    
    user_labels = user_labels_full[user_labels_full['region'] == region_map[region_name]].copy()
    user_labels = user_labels.dropna(subset=['I_am_working_in_field', 'gender', 'region'])
    user_labels.insert(0, 'user_id', user_labels.index)
    
    user_labels['I_am_working_in_field'] = (user_labels['I_am_working_in_field'] >= 0).astype(int)
    
    # Sensitive Attribute: Region
    le_region = LabelEncoder()
    sens_attr = torch.tensor(le_region.fit_transform(user_labels['region']), dtype=torch.long)
    # Task: Predict working field
    labels = torch.tensor(user_labels['I_am_working_in_field'].values, dtype=torch.long)
    
    le_age = LabelEncoder()
    node_features_df = user_labels[['completion_percentage', 'age', 'gender']].copy()
    node_features_df['age'] = le_age.fit_transform(node_features_df['age'])
    node_features = torch.tensor(node_features_df.values, dtype=torch.float)
    
    edge_index = _remap_edges(user_labels, user_edges)

    return Data(x=node_features, edge_index=edge_index, y=labels), sens_attr


def preprocess_data(data_dir, data_name, train_split, test_split=0.1):
    """
    Main function to load and preprocess a specified graph dataset.
    """
    dataset_loaders = {
        "credit": _load_credit_data,
        "german": _load_german_data,
        "bail": _load_bail_data,
        "nba": _load_nba_data,
        "pokec-z": lambda d: _load_pokec_data(d, 'pokec-z'),
        "pokec-n": lambda d: _load_pokec_data(d, 'pokec-n'),
    }

    if data_name not in dataset_loaders:
        raise ValueError(f"Invalid dataset name. Choose from: {list(dataset_loaders.keys())}")
    
    dataset_path = os.path.join(data_dir, data_name)
    if data_name.startswith('pokec'):
        dataset_path = os.path.join(data_dir, 'pokec')

    data, sens_attribute_tensor = dataset_loaders[data_name](dataset_path)

    data.train_mask, data.val_mask, data.test_mask = _create_masks(
        data.num_nodes, train_split, test_split
    )

    print(f"Successfully loaded and processed '{data_name}' dataset.")
    return data, sens_attribute_tensor