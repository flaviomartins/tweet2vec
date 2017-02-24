from __future__ import division
from builtins import object

import numpy as np
from scipy import special
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.preprocessing import normalize


class KulkarniKLDEuclideanDistances(object):

    def __init__(self, km):
        self.km = km

    def __call__(self, X, Y=None, Y_norm_squared=None, squared=False,
                 X_norm_squared=None):
        return kulkarni_kld_metric(X, Y, self.km)


def kulkarni_kld_metric(X, Y, km):
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
    kld_matrix : function
        Computes all pair-wise distances for a set of measurements
    entropy : function
        Computes entropy and K-L divergence
    """

    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, norm='l1', copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, norm='l1', copy=True)

    m = (X_normalized + Y_normalized)
    m /= 2.
    m = np.where(m, m, 1.)

    return 0.5*np.sum(special.xlogy(X_normalized, X_normalized/m) + special.xlogy(Y_normalized, Y_normalized/m), axis=1)