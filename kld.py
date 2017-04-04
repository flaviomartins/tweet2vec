from __future__ import division
from builtins import object

import numpy as np
from scipy.sparse import issparse
from scipy.special import rel_entr, xlogy
from sklearn.metrics.pairwise import check_pairwise_arrays, pairwise_distances
from sklearn.preprocessing import normalize


class KulkarniKLDEuclideanDistances(object):

    def __init__(self):
        pass

    def __call__(self, X, Y=None, Y_norm_squared=None, squared=False,
                 X_norm_squared=None):
        return pairwise_kld(X, Y)


def pairwise_kld(X, Y):
    X_dense = X
    if issparse(X):
        X_dense = X.todense()
    if X is Y:
        Y_dense = X_dense
    elif issparse(Y):
        Y_dense = Y.todense()
    else:
        Y_dense = Y

    return pairwise_distances(X_dense, Y_dense, metric=kld_metric)


def kld_metric(X, Y):
    """Compute Kulkarni's Negative Kullback-Liebler metric
    Parameters
    ----------
    a : array-like
        possibly unnormalized distribution.
    b : array-like
        possibly unnormalized distribution. Must be of same shape as ``a``.
    Returns
    -------
    j : float
    See Also
    --------
    entropy : function
        Computes entropy and K-L divergence
    """
    X, Y = check_pairwise_arrays(np.atleast_2d(X), np.atleast_2d(Y))

    X_normalized = normalize(X, norm='l1', copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, norm='l1', copy=True)

    XY_rel_entr = rel_entr(X_normalized, Y_normalized)
    YX_rel_entr = rel_entr(Y_normalized, X_normalized)

    distances = np.sum(XY_rel_entr + YX_rel_entr, axis=1)
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances
