import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import normalize, scale
from cvxopt import solvers, matrix
import scipy.io as scio
from utils import *
from measures import *
from model import *


def train(para):
    path = f'./dataset/{para.data}.mat'
    X = scio.loadmat(path)['X']
    labels = scio.loadmat(path)['Y'].squeeze()

    if para.If_scale == True:
        X = scale(X, axis=0)

    n, d = X.shape[0], X.shape[1]  

    hyper_edges = np.load(f"./pretrained/{para.data}_pretrain/hyper_edges.npy", allow_pickle=True)
    adj = np.hstack(hyper_edges) 
    init_graph = 1/para.num_bases * np.matmul(adj, adj.T) 
    pretrain_path = f'./pretrained/{para.data}_pretrain/pre_{para.selected_epoch}.pkl'

    input_dim = d
    model = my_Net(pretrain_path = pretrain_path,
                   num_features = input_dim,
                   hidden_size = para.hidden_size,
                   embedding_size = para.embedding_size,
                   alpha = para.alpha,
                   num_clusters = para.num_clusters,
                   v = 1 ).to(para.device)
    optimizer = Adam(model.parameters(),
                     lr=para.lr, weight_decay=para.weight_decay)

    adj = torch.Tensor(adj).to(para.device)
    adj_label = torch.Tensor(init_graph).to(para.device)
    X = torch.Tensor(X.astype(np.float32)).to(para.device)  
    y = labels

    with torch.no_grad():
        _, pretrain_z = model.gat(X, adj)

    kmeans = KMeans(n_clusters = para.num_clusters, n_init=20)
    y_pred = kmeans.fit_predict(pretrain_z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(para.device)

    S = Generate_S(hyper_edges)
    Phi = Generate_Phi(S) + para.weight*np.eye(para.num_bases)
    G = -1*np.eye(para.num_bases)
    h = np.zeros(para.num_bases)
    a = np.ones(para.num_bases).reshape(-1, 1)

    Phi = matrix(Phi)
    G = matrix(G)
    h = matrix(h)
    a = matrix(a).T 
    b = matrix(1.)

    for epoch in range(para.max_epoch):
        model.train()
        if epoch % para.update_interval == 0:
            # update_interval
            A_pred, z, Q = model(X, adj)

            pred_q = Q.detach().data.cpu().numpy().argmax(1)  
            acc, nmi, ari, f1 = eva(y, pred_q, epoch)

        if epoch % (para.update_interval_2) == 0:
            temp_A = A_pred.detach().data.cpu().numpy()
            eta = Generate_eta(S, temp_A)
            eta = matrix(eta)
        
            solvers.options['show_progress'] = False
            sol = solvers.qp(Phi,eta,G,h,a,b)
            gamma = np.array(sol['x'])
            adj_label = Refine(S, gamma, n)
            adj_label = torch.Tensor(adj_label).to(para.device)

        A_pred, z, q = model(X, adj)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = para.lam*kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"final result: {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")

