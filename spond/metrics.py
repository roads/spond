"""Evaluation metrics."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import utils
import scipy as sp
from scipy import stats


# TODO: refactor
def mapping_accuracy(f_x, y):
    """Compute mapping accuracy.
    Assumes inputs f_x and y are aligned.
    """
    n_concept = f_x.shape[0]
    n_half = int(np.ceil(n_concept / 2))

    # Create nearest neighbor graph for y.
    neigh = NearestNeighbors(n_neighbors=n_half)
    neigh.fit(y)
    # Determine which concepts of y are closest for each point in f_x.
    _, indices = neigh.kneighbors(f_x)

    dmy_idx = np.arange(n_concept)
    dmy_idx = np.expand_dims(dmy_idx, axis=1)

    locs = np.equal(indices, dmy_idx)

    is_correct_half = np.sum(locs[:, 0:n_half], axis=1)
    is_correct_10 = np.sum(locs[:, 0:10], axis=1)
    is_correct_5 = np.sum(locs[:, 0:5], axis=1)
    is_correct_1 = locs[:, 0]

    acc_half = np.mean(is_correct_half)
    acc_10 = np.mean(is_correct_10)
    acc_5 = np.mean(is_correct_5)
    acc_1 = np.mean(is_correct_1)
    return acc_1, acc_5, acc_10, acc_half


def alignment_correlation(systemA, systemB, f=None):

    """
    Calculate alignment correlation between two systems with mapping in 
    both directions (A->B and B->A)

    Assumes systems are in the same space
    """

    def f(x, y):
        return np.sqrt(np.sum((x-y)**2, axis=1))

    # Index of upper triangular matrices
    idx_upper = np.triu_indices(systemA.shape[0], 1)

    # Pairwise distance matrix between system A and system B
    pairwise_both = utils.pairwise_distance(systemA, systemB)

    # Argmin by row (ie system A) and column (ie system B)
    row_argmin = np.argmin(pairwise_both, 1)
    col_argmin = np.argmin(pairwise_both, 0)

    # Create a mapping for A->B and B->A, arrange matrices accordingly
    systemB_Aorg = systemB[row_argmin, :]
    systemA_Borg = systemA[col_argmin, :]

    # Take upper diagonal of corresponding sim matrices for A->B
    vec_A = f(systemA[idx_upper[0]], systemA[idx_upper[1]])
    vec_B_Aorg = f(systemB_Aorg[idx_upper[0]], systemB_Aorg[idx_upper[1]])

    # Spearman correlation
    r_s1 = sp.stats.spearmanr(vec_A, vec_B_Aorg)[0]

    # Repeat for B->A mapping
    vec_B = f(systemB[idx_upper[0]], systemB[idx_upper[1]])
    vec_A_Borg = f(systemA_Borg[idx_upper[0]], systemA_Borg[idx_upper[1]])

    ### something in here is throwing 'divide by zero or nan' errors
    r_s2 = sp.stats.spearmanr(vec_B, vec_A_Borg)[0]

    r_s = (r_s1 + r_s2)/2

    return r_s