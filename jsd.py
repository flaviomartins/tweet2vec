from __future__ import division
from builtins import range

import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_array_almost_equal
from scipy import special, stats
from sklearn.metrics.pairwise import check_pairwise_arrays, pairwise_distances
from sklearn.preprocessing import normalize


# adapted from gh:luispedro/scipy
def jensen_shannon_divergence(X, Y):
    """Compute Jensen-Shannon Divergence
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
    jsd_matrix : function
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


def pairwise_entropy(X, Y):
    return pairwise_distances(X, Y, metric=stats.entropy)


def test_jensen_shannon_divergence():
    for _ in range(8):
        a = np.random.random((1, 16))
        b = np.random.random((1, 16))
        c = a+b

        assert_(jensen_shannon_divergence(a, a) < 1e-4)
        assert_(jensen_shannon_divergence(a, b) > 0.)
        assert_(jensen_shannon_divergence(a, b) >
                jensen_shannon_divergence(a, c))
        assert_array_almost_equal(jensen_shannon_divergence(a, b),
                                  jensen_shannon_divergence(a, b*6))

    a = np.array([[1, 0, 0, 0]], float)
    b = np.array([[0, 1, 0, 0]], float)
    assert_almost_equal(jensen_shannon_divergence(a, b), np.log(2))

    a = np.array([[1, 0, 0, 0]], float)
    b = np.array([[1, 1, 1, 1]], float)
    m = a/a.sum() + b/b.sum()
    m = (pairwise_entropy(a, m) + pairwise_entropy(b, m)) / 2
    expected = m.ravel()
    assert_almost_equal(jensen_shannon_divergence(a, b), expected)

    a = np.random.random((4, 16))
    b = np.random.random((4, 16))
    direct = jensen_shannon_divergence(a, b)
    # indirect = np.array([jensen_shannon_divergence(aa, bb)
    #                      for aa, bb in zip(a, b)])
    # assert_array_almost_equal(direct, indirect)


if __name__ == '__main__':
    test_jensen_shannon_divergence()
