# GNN’s FAME: Fairness-Aware MEssages for Graph Neural Networks

This repository contains the code and experiments for the paper "GNN’s FAME: Fairness-Aware MEssages for Graph Neural Networks".
The primary contribution of this work is the development of two novel in-processing and GNN-specific bias mitigation approaches, namely **FAME** (Fairness-Aware MEssages) and its variant **A-FAME** (Attention-based Fairness-Aware MEssages), designed for GCN-based and GAT-based architectures, respectively.

## Table of Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)

## Introduction

Graph Neural Networks (GNNs) are powerful tools for learning representations of graph-structured data. However, GNNs are susceptible to biases that can arise from the underlying data, leading to unfair predictions. To address this issue, we propose two novel message-passing layers:

- **FAME (Fairness-Aware Message Passing)**: This layer adjusts the messages during the aggregation phase based on the disparities in sensitive attributes of connected nodes.
- **A-FAME (Attention Fairness-Aware Message Passing)**: This layer extends FAME by incorporating an attention mechanism to weigh the importance of node connections dynamically.

Note: enhanced_fame is a WIP extension of this repository.

These layers aim to ensure more equitable outcomes by mitigating bias propagation within GNNs.

## Repository Structure
The repository contains:
* *Dataset*: Directory containing German, Credit, and Bail datasets.
* *calculate_fairness.py*: File containing the implementation of the fairness metrics calculation.
* *fame.py*: File containing the implementation of the FAME and A-FAME layers.
* *main.py*: File containing the implementation of the training and evaluation of the GNN models.
* *model.py*: File containing the implementation of the GNN models (GCN, GAT, FAME, A-FAME).
* *preprocess_data.py*: File containing the implementation of the data preprocessing.
* *set_uid.py*: File containing the implementation of the unique identifier setting for each node in the dataset.
* *utils.py*: File containing the implementation of the utility functions.

## Datasets
Extract the datasets from the datasets folder.  
Run the file `preprocess_data.py` to preprocess the datasets first, then run file `set_uid.py` to set the unique identifier for each node in the dataset.  
The datasets adopted in the paper's evaluation can be found at the following links (or in the datasets folder - in the case of German and Credit):
- [German](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- [Credit](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [Pokec](https://snap.stanford.edu/data/soc-Pokec.html)
