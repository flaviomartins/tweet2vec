from __future__ import division
from builtins import range

import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_array_almost_equal
from scipy.special import rel_entr
from scipy.stats import entropy
from scipy.sparse import issparse
from sklearn.metrics.pairwise import check_pairwise_arrays, pairwise_distances
from sklearn.preprocessing import normalize


def pairwise_jsd(X, Y):
    X_dense = X
    if issparse(X):
        X_dense = X.todense()
    if X is Y:
        Y_dense = X_dense
    elif issparse(Y):
        Y_dense = Y.todense()
    else:
        Y_dense = Y

    return pairwise_distances(X_dense, Y_dense, metric=jensen_shannon_divergence)


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
    entropy : function
        Computes entropy and K-L divergence
    """
    X, Y = check_pairwise_arrays(np.atleast_2d(X), np.atleast_2d(Y))

    X_normalized = normalize(X, norm='l1', copy=True)

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, norm='l1', copy=True)

    m = (X_normalized + Y_normalized)
    m /= 2.
    m = np.where(m, m, 1.)

    XXm_rel_entr = rel_entr(X_normalized, m)
    YYm_rel_entr = rel_entr(Y_normalized, m)

    distances = 0.5 * np.sum(XXm_rel_entr + YYm_rel_entr, axis=1)
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances


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

    a = np.array([[1, 0, 0, 0]])
    b = np.array([[0, 1, 0, 0]])
    assert_almost_equal(jensen_shannon_divergence(a, b), np.log(2))

    a = np.array([1, 0, 0, 0], float)
    b = np.array([1, 1, 1, 1], float)
    m = a/a.sum() + b/b.sum()
    expected = (entropy(a, m) + entropy(b, m)) / 2
    calculated = jensen_shannon_divergence(a, b)
    assert_almost_equal(calculated, expected)

    a = np.random.random((2, 12))
    b = np.random.random((10, 12))
    direct = pairwise_jsd(a, b)
    indirect = pairwise_distances(a, b, metric=jensen_shannon_divergence)
    assert_array_almost_equal(direct, indirect)


if __name__ == '__main__':
    test_jensen_shannon_divergence()
