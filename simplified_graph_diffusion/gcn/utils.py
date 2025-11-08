# utils for gcn classification.
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import networkx as nx
import pickle as pkl
import random as rd

def load_data(dataset, task='link_prediction', feat_norm=True):
    if task == 'link_prediction':
        names = ['graph', 'feature']
        objects = []
        for i in range(len(names)):
            with open("data/{}.{}".format(dataset, names[i]), 'rb') as f:
                objects.append(pkl.load(f))
        adj = objects[0]
        if feat_norm == True:
            feature = preprocess_feature(objects[1])
        else:
            feature = torch.FloatTensor(np.array(objects[1].todense()))
        return adj, feature

    elif task == 'classification':
        names = ['graph', 'feature','labels']
        objects = []
        for i in range(len(names)):
            with open("data/{}.{}".format(dataset, names[i]), 'rb') as f:
                objects.append(pkl.load(f))
        adj = objects[0]
        if feat_norm == True:
            feature = preprocess_feature(objects[1])
        else:
            feature = torch.FloatTensor(np.array(objects[1].todense()))
        labels = labels_encode(objects[2])
        return adj, feature, labels

def preprocess_feature(feature):
    rowsum = np.array(feature.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    feature = r_mat_inv.dot(feature)
    feature = torch.FloatTensor(np.array(feature.todense()))
    return feature

def sparse_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_torch_sparse_tensor(adj_normalized)

def preprocess_graph_diff(adj, diff_n, diff_alpha):
    adj = sp.coo_matrix(adj)
    adj_ = propagation_prob(adj, diff_alpha)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_diff = np.power(adj_normalized, diff_n)
    # adj_diff = adj_normalized + np.power(adj_normalized, diff_n)
    return sparse_to_torch_sparse_tensor(adj_diff)

def preprocess_graph_diff_nth(adj, n_diff, alpha):
    adj_diff = sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    adj_ = propagation_prob(adj, alpha)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    for i in range(1, n_diff+1):
        adj_diff += np.power(adj_normalized, i)
    return sparse_to_torch_sparse_tensor(adj_diff)

def propagation_prob(adj, diff_alpha):
    if  diff_alpha != 0.5: 
        adj = (diff_alpha) * adj + (1-diff_alpha) * sp.eye(adj.shape[0])
    else:
        adj = adj + sp.eye(adj.shape[0])
    return adj

def labels_encode(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    labels = torch.LongTensor(np.where(labels_onehot)[1])
    return labels

def split(data_len: int, train: int, val: int, test: int):
    idx_train = rd.sample(range(data_len), train); remain_1= [i for i in range(data_len) if i not in idx_train]
    idx_val = rd.sample(remain_1, val);            remain_2= [i for i in range(data_len) if i not in idx_train+idx_val]
    idx_test = rd.sample(remain_2, test)
    return idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def process_closed_form_graph_diff(adj, diff_n, diff_alpha):
    """Compute I + S + ... + S^n using closed-form geometric series."""
    if diff_n < 0:
        raise ValueError("diff_n must be non-negative")

    adj = sp.coo_matrix(adj)
    adj_ = propagation_prob(adj, diff_alpha)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocsr()

    identity = sp.eye(adj_normalized.shape[0], format='csr')
    if diff_n == 0:
        return sparse_to_torch_sparse_tensor(identity)

    geom_numerator = identity - (adj_normalized ** (diff_n + 1))
    geom_denominator = identity - adj_normalized
    rhs = geom_numerator.todense()
    closed_form_dense = spsolve(geom_denominator.tocsc(), rhs)
    closed_form_sparse = sp.csr_matrix(closed_form_dense)
    return sparse_to_torch_sparse_tensor(closed_form_sparse.tocoo())

def process_sumpow_graph_diff(adj, diff_n, diff_alpha):
    """Compute I + S + ... + S^n with explicit iterative accumulation."""
    if diff_n < 0:
        raise ValueError("diff_n must be non-negative")

    adj = sp.coo_matrix(adj)
    adj_ = propagation_prob(adj, diff_alpha)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocsr()

    identity = sp.eye(adj_normalized.shape[0], format='csr')
    cumulative = identity.copy()
    current_power = identity.copy()

    for _ in range(1, diff_n + 1):
        current_power = current_power.dot(adj_normalized)
        cumulative = cumulative + current_power

    return sparse_to_torch_sparse_tensor(cumulative.tocoo())
