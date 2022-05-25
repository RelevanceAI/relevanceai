import numpy as np

from relevanceai.utils.integration_checks import is_scipy_available
from relevanceai.utils.decorators.analytics import track


@track
def euclidean_distance_matrix(a, b, decimal=None):
    A = np.array(a)
    B = np.array(b)
    M = A.shape[0]
    N = B.shape[0]
    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones((1, M))
    B_dots = (B * B).sum(axis=1).reshape((N, 1)) * np.ones((1, N))
    dist = A_dots + B_dots - 2 * A.dot(B.T)
    dist[dist < 1e-9] = 0
    dist = np.sqrt(dist + dist.T)
    if decimal:
        dist = np.around(dist, decimal)
    return dist.tolist()
