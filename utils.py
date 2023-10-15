import torch
import numpy as np


def Indicator(labels, n_clusters):
    n = len(labels)
    ind = np.zeros((n, n_clusters), dtype=int)
    ind[np.arange(n), labels] = 1
    return ind


def adj_to_hyper(adj):
    device = adj.device
    adj.fill_diagonal_(0) 

    edge_index = torch.nonzero(adj)

    n = adj.size(0) 
    num_edges = edge_index.size(0) 

    edge_index = edge_index.reshape(-1)

    a = torch.arange(num_edges).repeat_interleave(2, dim=0)  
    a = a.to(device)  

    # 生成要填充的点
    index = torch.cat([edge_index.unsqueeze(0), a.unsqueeze(0)], dim = 0)
    index = torch.transpose(index, 0, 1)

    hyper_adj = torch.zeros([n,num_edges])
    flat_index = index[:, 0] * hyper_adj.shape[1] + index[:, 1]
    hyper_adj.view(-1)[flat_index] = 1

    # self-loop
    I = torch.eye(n)
    hyper_adj = torch.cat((hyper_adj, I), dim=1)
    return hyper_adj.to(device)


def Generate_S(hyper_edges):
    M = hyper_edges.shape[0]
    S = np.empty(M, dtype=object)
    for i in range(M):
        S[i] = np.dot(hyper_edges[i], hyper_edges[i].T)

    return S


def Generate_Phi(S):
    M = S.shape[0]
    Phi = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            Phi[i, j] = np.sum(S[i] * S[j])

    return Phi


def Generate_eta(S, temp_A):
    M = S.shape[0]
    eta = np.zeros(M)
    for i in range(M):
        eta[i] = np.sum(S[i] * temp_A)

    return eta


def Refine(S, gamma, n):
    M = S.shape[0]
    adj_label = np.zeros((n, n))

    for i in range(M):
        adj_label += gamma[i]*S[i]

    return adj_label