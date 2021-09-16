# This module contains similarity functions
# All of them should take 2 vectors of embeddings
# and compute the pairwise similarity
# None of these are GPU-ised.
import numpy as np


from scipy.spatial.distance import cdist


def exponential(x1, x2, decay=1):
    # numpy, so both must be on CPU
    n1, d1 = x1.shape
    n2, d2 = x2.shape
    assert d1 == d2, "x1 and x2 must have same dimension"
    out = np.empty((n1, n2))
    for j, x11 in enumerate(x1):
        out[j] = np.exp(decay * -cdist([x11], x2, 'euclidean'))
    return out


def dot(x1, x2):
    # numpy, so both must be on CPU
    n1, d1 = x1.shape
    n2, d2 = x2.shape
    assert d1 == d2, "x1 and x2 must have same dimension"
    out = np.empty((n1, n2))
    for j, x11 in enumerate(x1):
        out[j] = np.einsum('j,ij->i', x11, x2)
    return out


def cosine(x1, x2):
    # numpy, so both must be on CPU
    n1, d1 = x1.shape
    n2, d2 = x2.shape
    assert d1 == d2, "x1 and x2 must have same dimension"
    out = 1 - cdist(x1, x2, metric='cosine')
    return out
