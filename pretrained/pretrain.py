import os
import numpy as np
from sklearn.cluster import KMeans
from torch.optim import Adam
from utils import *
from sklearn.preprocessing import normalize, scale
from measures import *
from model import *
import scipy.io as scio


def pretrain(para):
    save_path = f"./{para.data}_pretrain"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    path = f'./dataset/{para.data}.mat' 
    X = scio.loadmat(path)['X']
    labels = scio.loadmat(path)['Y'].squeeze()
    print(X.shape, labels.shape)

    if para.If_scale == True:
        X = scale(X, axis=0) 

    n, d = X.shape[0], X.shape[1] 

    kmeans = KMeans(n_clusters=para.num_clusters,  n_init=10) 
    kmeans.fit(X) 
    eva(labels, kmeans.labels_, 'base clustering')

    hyper_edges = np.empty(para.num_bases, dtype=object)

    for i in range(para.num_bases):
        n_clusters = np.random.randint(6*para.num_clusters, 8*para.num_clusters) 
        kmeans = KMeans(n_clusters=n_clusters, n_init = 10)  
        kmeans.fit(X)  
        ind = Indicator(kmeans.labels_, n_clusters)  
        hyper_edges[i] = ind

    np.save(f"./{para.data}_pretrain/hyper_edges.npy", hyper_edges)

    adj = np.hstack(hyper_edges) 
    init_graph = 1/para.num_bases * np.matmul(adj, adj.T)

    input_dim = d
    model = hyper_GAT(
            num_features=input_dim,
            hidden_size=para.hidden_size,
            embedding_size=para.embedding_size,
            alpha=para.alpha,
        ).to(para.device)
    print(model)
    optimizer = Adam(model.parameters(), lr=para.lr, weight_decay=para.weight_decay)

    adj = torch.Tensor(adj).to(para.device) 
    adj_label = torch.Tensor(init_graph).to(para.device)
    X = torch.Tensor(X.astype(np.float32)).to(para.device) 
    y = labels

    for epoch in range(para.max_epoch+1):
        model.train()
        A_pred, z = model(X, adj)

        with torch.no_grad():
            _, z = model(X, adj)
            kmeans = KMeans(n_clusters = para.num_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)

        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        print(f"re_loss {loss:.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"./{para.data}_pretrain/pre_{epoch}.pkl")