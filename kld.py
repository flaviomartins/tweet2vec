from __future__ import division

import numpy as np
from scipy.sparse import issparse
from scipy.spatial.distance import _copy_array_if_base_present
from scipy.special import rel_entr, xlogy
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.preprocessing import normalize


def cdist_kld_sparse(X, Y, p_B, **kwargs):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
    # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist_kld(X, Y, p_B, **kwargs)
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist_kld(x.todense(), Y, p_B, **kwargs) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist_kld(X, y.todense(), p_B, **kwargs) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist_kld(x.todense(), y.todense(), p_B, **kwargs) [0]
    return d


def cdist_kld(XA, XB, pB=None, a=0.1):
    # You can also call this as:
    #     Y = cdist(XA, XB, 'test_abc')
    # where 'abc' is the metric being tested.  This computes the distance
    # between all pairs of vectors in XA and XB using the distance metric 'abc'
    # but with a more succinct, verifiable, but less efficient implementation.

    # Store input arguments to check whether we can modify later.
    input_XA, input_XB = XA, XB

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')

    # The C code doesn't do striding.
    XA = _copy_array_if_base_present(XA)
    XB = _copy_array_if_base_present(XB)

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]
    dm = np.zeros((mA, mB), dtype=np.double)
    for i in xrange(0, mA):
        for j in xrange(0, mB):
            dm[i, j] = kld_metric(XA[i, :], XB[j, :], pB, a)

    return dm


def kld_metric(X, Y, p_B, a=0.1, **kwargs):
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

    a_p_B = a*p_B
    p_D = (1-a)*X_normalized + a_p_B
    p_C = Y_normalized

    m = (p_C + p_D)
    m /= 2.
    m = np.where(m, m, 1.)

    XXm_rel_entr = rel_entr(p_C, m)
    YYm_rel_entr = rel_entr(p_D, m)

    distances = 0.5 * np.sum(XXm_rel_entr + YYm_rel_entr, axis=1)
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances
