import numpy as np
from sklearn.neighbors import NearestNeighbors

def chamfer_distance(a, b):
    nn_a = NearestNeighbors(n_neighbors=1).fit(b)
    d_a, _ = nn_a.kneighbors(a)
    nn_b = NearestNeighbors(n_neighbors=1).fit(a)
    d_b, _ = nn_b.kneighbors(b)
    return float(d_a.mean() + d_b.mean())

def f_score(a, b, tau=0.01):
    nn_a = NearestNeighbors(n_neighbors=1).fit(b)
    d_a, _ = nn_a.kneighbors(a)
    nn_b = NearestNeighbors(n_neighbors=1).fit(a)
    d_b, _ = nn_b.kneighbors(b)
    precision = float((d_a[:,0] < tau).mean())
    recall = float((d_b[:,0] < tau).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)