import torch
import numpy as np
import random
import argparse
from utils import *
from train import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='vehicle_uni')
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lam', type=float, default=1000)
    parser.add_argument('--If_scale', default=True)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--update_interval', type=int, default=3)
    parser.add_argument('--update_interval_2', type=int, default=5)
    parser.add_argument('--weight', type=int, default=1e-2)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--seed', type=int, default=0)
    para = parser.parse_args()  

    para.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    para.num_bases = 20  

    torch.cuda.set_device(0)  

    para.data = 'MSRA25_uni'
    if para.data == 'MSRA25_uni': 
        para.num_clusters = 12
        para.hidden_size = 50
        para.embedding_size = 30
    elif para.data == 'ORL':
        para.num_clusters = 40
        para.hidden_size = 300
        para.embedding_size = 100
        para.lr = 0.001
        para.seed = 1
    elif para.data == 'USPSdata_uni':
        para.num_clusters = 10
        para.hidden_size = 64
        para.embedding_size = 16
        para.lr = 0.005
    elif para.data == 'Yale_32x32':
        para.num_clusters = 38
        para.hidden_size = 200
        para.embedding_size = 100
        para.lr = 0.0005  
        para.lam = 0.1
    elif para.data == 'Isolet':
        para.num_clusters = 26
        para.hidden_size = 300
        para.embedding_size = 200
    elif para.data == 'TOX_171':
        para.num_clusters = 4
        para.hidden_size = 300
        para.embedding_size = 100
        para.If_scale = False
    elif para.data == 'arcene':  
        para.num_clusters = 2
        para.hidden_size = 200
        para.embedding_size = 100
    elif para.data == 'letter_uni':
        para.num_clusters = 26
        para.hidden_size = 50
        para.embedding_size = 20
        para.If_scale = False
    elif para.data == 'segment_uni':
        para.num_clusters = 7
        para.hidden_size = 10
        para.embedding_size = 10
        para.If_scale = False
    elif para.data == 'vote_uni':
        para.num_clusters = 2
        para.hidden_size = 10
        para.embedding_size = 5
    elif para.data == 'vehicle_uni':  
        para.num_clusters = 4
        para.hidden_size = 10
        para.embedding_size = 5
    else:
        raise NotImplementedError("Unexpected Dataset")
    

    para.selected_epoch = 45
    para.max_epoch = 50

    np.random.seed(para.seed)
    torch.manual_seed(para.seed)
    random.seed(para.seed)
    torch.cuda.manual_seed(para.seed)

    train(para)












