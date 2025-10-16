import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from utils.math_utils import z_score


def distance_to_weight(W, sigma2=0.1, epsilon=0.5, gat_version=False):
    """Compute weight matrix from distance matrix."""
    n = W.shape[0]
    W = W / 10000.
    W2 = W * W
    W_mask = np.ones([n, n]) - np.identity(n)

    W = np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

    if gat_version:
        W[W > 0] = 1
        W += np.identity(n)

    return W


def make_dataset(config, W):
    """Return train, val, test datasets as lists of Data objects."""
    # Load data
    data = pd.read_csv('./dataset/PeMSD7_V_228.csv', header=None).values
    mean, std = np.mean(data), np.std(data)
    data = z_score(data, mean, std)

    n_node = data.shape[1]
    n_window = config['N_PRED'] + config['N_HIST']

    # Build edge index/attr
    edge_index, edge_attr = [], []
    for i in range(n_node):
        for j in range(n_node):
            if W[i, j] != 0:
                edge_index.append([i, j])
                edge_attr.append([W[i, j]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Build graph sequences
    sequences = []
    for i in range(config['N_DAYS']):
        for j in range(config['N_SLOT']):
            g = Data()
            g.__num_nodes__ = n_node
            g.edge_index = edge_index
            g.edge_attr = edge_attr

            sta = i * config['N_DAY_SLOT'] + j
            end = sta + n_window
            full_window = np.swapaxes(data[sta:end, :], 0, 1)

            g.x = torch.FloatTensor(full_window[:, :config['N_HIST']])
            g.y = torch.FloatTensor(full_window[:, config['N_HIST']:])
            sequences.append(g)

    # Split into train/val/test
    split_train, split_val, split_test = config['SPLITS']
    n_slot = config['N_SLOT']
    i = n_slot * split_train
    j = n_slot * split_val

    train = sequences[:i]
    val = sequences[i:i + j]
    test = sequences[i + j:]

    return train, val, test, mean, std, n_node
