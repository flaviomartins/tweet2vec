from __future__ import division
from builtins import range

import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_array_almost_equal
from scipy.spatial.distance import cdist
from scipy.special import rel_entr, xlogy
from scipy.stats import entropy
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
    entropy : function
        Computes entropy and K-L divergence
    """
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)
    m = X + Y
    m /= 2.
    return 0.5 * np.sum(rel_entr(X, m) + rel_entr(Y, m), axis=1)


def test_jensen_shannon_divergence():
    for _ in range(8):
        a = normalize(np.random.random((1, 16)), norm='l1', copy=True)
        b = normalize(np.random.random((1, 16)), norm='l1', copy=True)
        c = normalize(a + b, norm='l1', copy=True)

        assert_(jensen_shannon_divergence(a, a) < 1e-4)
        assert_(jensen_shannon_divergence(a, b) > 0.)
        assert_(jensen_shannon_divergence(a, b) >
                jensen_shannon_divergence(a, c))
        assert_array_almost_equal(jensen_shannon_divergence(a, b),
                                  jensen_shannon_divergence(a, normalize(b * 6, norm='l1', copy=True)))

    a = np.array([[1, 0, 0, 0]])
    b = np.array([[0, 1, 0, 0]])
    assert_almost_equal(jensen_shannon_divergence(a, b), np.log(2))

    a = np.array([1, 0, 0, 0], float)
    b = np.array([1, 1, 1, 1], float)
    m = a / a.sum() + b / b.sum()
    expected = (entropy(a, m) + entropy(b, m)) / 2
    calculated = jensen_shannon_divergence(a, normalize(np.atleast_2d(b), norm='l1', copy=True))
    assert_almost_equal(calculated, expected)

    a = np.random.random((1, 12))
    b = np.random.random((10, 12))
    direct = jensen_shannon_divergence(a, b)
    indirect = cdist(a, b, metric=jensen_shannon_divergence)[0]
    assert_array_almost_equal(direct, indirect)


if __name__ == '__main__':
    test_jensen_shannon_divergence()
