from __future__ import print_function
import numpy as np


# WT is a pandas dataframe WT, in which row_i is the
# vector embedding of word_i and is also indexed by word_i.
# Discluding the index WT is nxD.
# WTn is nx1 and stores the 2-norm of each row vector in WT.
# vj is an 1xD word embedding.
# Returns a nx1 vector where n is the number of embeddings.
# The delta function has been evaluated on all w_i.
def delta(WT, WTn, vj):
    # WTwj is nx1
    WTwj = np.dot(WT, vj)
    # jnorm is as scalar
    jnorm = np.linalg.norm(vj)
    # denom is nx1
    denom = (WTn * jnorm)**-1.0
    # entry wise multiplication of vectors
    # like dot product but without summing
    return np.multiply(WTwj, denom)


# WT is a pandas dataframe WT, in which row_i is the
# vector embedding of word_i and is also indexed by word_i.
# Discluding the index WT is nxD.
# WTn is nx1 and stores the 2-norm of each row vector in WT.
# If y is specified return the single distance measure for that y.
# Otherwise return the word with the greatest distance measure.
def cosadd(WT, WTn, a, b, x, y=None):
    # wx_m_wa_p_wb is 1xD
    wx_m_wa_p_wb = WT.loc[x] - WT.loc[a] + WT.loc[b]
    # d is nx1
    d = delta(WT, WTn, wx_m_wa_p_wb)
    # if y is specified then return that correspondig distance measure
    if y:
        return d[np.where(WT.index == y)[0][0]]
    # zero distances corresponding to the inputs
    for i in [a, b, x]:
        d[np.where(WT.index == i)[0][0]] = 0.0
    # find the index of the maximum delta
    max_idx = np.argmax(d)
    return WT.index[max_idx]


# WT is a pandas dataframe WT, in which row_i is the
# vector embedding of word_i and is also indexed by word_i.
# Discluding the index WT is nxD.
# WTn is nx1 and stores the 2-norm of each row vector in WT.
# If y is specified return the single distance measure for that y.
# Otherwise return the word with the greatest distance measure.
def cosmult(WT, WTn, a, b, x, e=0.001, y=None):
    # dy[abx] are nx1
    dya = (delta(WT, WTn, WT.loc[a]) + 1.0) / 2.0
    dyb = (delta(WT, WTn, WT.loc[b]) + 1.0) / 2.0
    dyx = (delta(WT, WTn, WT.loc[x]) + 1.0) / 2.0
    # entry wise multiplication of vectors
    # dybdyx is nx1
    dybdyx = np.multiply(dyb, dyx)
    # dyae is nx1
    dyae = dya + e
    # measures is nx1
    measures = np.multiply(dybdyx, dyae**-1.0)
    # if y is specified then return that correspondig distance measure
    if y:
        return measures[np.where(WT.index == y)[0][0]]
    # find the index of the maximum measure
    max_idx = np.argmax(measures)
    return WT.index[max_idx]
