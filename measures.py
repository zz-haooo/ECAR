import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix


def Adjust_Label(arr):
    if np.any(arr == 0): return arr
    else: return arr - 1


def Cluster_Map(arr):
    unique_elements = np.unique(arr)
    mapping_dict = {val: i for i, val in enumerate(unique_elements)}
    mapped_arr = np.array([mapping_dict[val] for val in arr])
    return mapped_arr


def cluster_acc(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "shape incompatibility"

    Y_true = Adjust_Label(y_true)
    Y_pred = Cluster_Map(y_pred)
    n_clusters = len(np.unique(Y_pred))
    n_classes = len(np.unique(Y_true))

    conf_matrix = np.zeros((n_classes, n_clusters))

    for i in range(len(Y_true)):
        conf_matrix[Y_true[i], Y_pred[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    n_correct = conf_matrix[row_ind, col_ind].sum()

    acc = n_correct / len(y_true)
    return acc


def purity(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "shape incompatibility"

    y = Adjust_Label(y_true)
    contingency = contingency_matrix(y, y_pred)
    purity = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)
    return purity


def eva(y_true, y_pred, epoch='0'):
    acc = cluster_acc(y_true, y_pred)
    pur = purity(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    print(f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, pur {pur:.4f}")
    return acc, nmi, ari, pur
