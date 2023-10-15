import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import *


class my_Net(nn.Module):
    def __init__(self, pretrain_path, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(my_Net, self).__init__()
        self.num_clusters = num_clusters
        self.v = v
        self.gat = hyper_GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(pretrain_path, map_location='cpu'), strict=False,)

        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj):
        A_pred, z = self.gat(x, adj)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class hyper_GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(hyper_GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = hyper_GATLayer(num_features, hidden_size, alpha)
        self.conv2 = hyper_GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj):
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)

        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class hyper_GATLayer(nn.Module):

    def __init__(self, in_features, out_features, alpha=0.2):
        super(hyper_GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):

        deg_adj = torch.pow(adj.sum(dim = 0), -1)
        deg_adj = torch.diag_embed(deg_adj).to(adj.device)  

        H = torch.mm(input, self.W)  

        WHE = torch.mm(torch.mm(deg_adj, adj.t()), H)  

        attn_for_self = torch.mm(H, self.a_self)  # (N,1)

        attn_for_neighs = torch.mm(WHE, self.a_neighs)  # (K,1)
        attn_dense = attn_for_self + attn_for_neighs.t()  # (N,K)

        attn_dense = self.leakyrelu(attn_dense)  # (N,K)

        zero_mat = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_mat)
        attention = F.softmax(adj, dim=1) 
        H_out = 0.5*torch.mm(attention, WHE) + 0.5*H

        return H_out

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )
