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

These layers aim to ensure more equitable outcomes by mitigating bias propagation within GNNs.

## Repository Structure
The repository contains two folders:
* *Layers*: source code of the proposed FAME and A-FAME layers.
* *Experiments*: notebooks with code used in the paper's experimental setup.

## Datasets
The datasets adopted in the paper's evaluation can be found at the following links:
- [German](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- [Credit](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [Pokec](https://snap.stanford.edu/data/soc-Pokec.html)
