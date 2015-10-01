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
    # norms is nx1
    # norms = np.linalg.norm(WT, axis=1)
    # jnorm is as scalar
    jnorm = np.linalg.norm(vj)
    # denom is nx1
    # denom = (norms * jnorm)**-1.0
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
    # dya = delta(WT, WT.loc[a])
    # dyb = delta(WT, WT.loc[b])
    # dyx = delta(WT, WT.loc[x])
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


def olddelta(i, j):
    return np.dot(i, j) / (np.dot(i, i) * np.dot(j, j))**0.5


def oldcosadd(WT, a, b, x):
    wx_m_wa_p_wb = WT.loc[x] - WT.loc[a] + WT.loc[b]

    def dm(y):
        return olddelta(WT.iloc[y], wx_m_wa_p_wb)
    return dm


def oldcosmult(WT, a, b, x, e):
    wa = WT.loc[a]
    wb = WT.loc[b]
    wx = WT.loc[x]

    def dm(y):
        wy = WT.iloc[y]
        return olddelta(wy, wb) * olddelta(wy, wx) / (olddelta(wy, wa) + e)
    return dm


def bruteforce(WT, dm):
    # max_delta = 0.0
    max_delta = float('-inf')
    max_idx = None
    for i in xrange(WT.shape[0]):
        cur_delta = dm(i)
        # print('delta({:s},{:s},{:s},{:s})={:0.2f}'.format(
        #     a, b, x, WT.index[i], cur_delta))
        # if abs(cur_delta) > max_delta:
        if cur_delta > max_delta:
            if max_idx:
                print('{:s} > {:s}'.format(WT.index[i], WT.index[max_idx]))
            max_delta = cur_delta
            max_idx = i
    return WT.index[max_idx]
