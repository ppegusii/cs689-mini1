from __future__ import print_function
import numpy as np
import pandas as pd


# Returns the file parsed to a pandas dataframe WT, in which row_i is the
# vector embedding of word_i and is also indexed by word_i.
def parse(fileName):
    # Note quotechar is the same as the delimiter.
    # This is required because read_csv defaults to using '"' as the
    # quotechar, which causes an EOF error because '"' is a word.
    # Work around given https://github.com/pydata/pandas/issues/5500
    return pd.read_csv(fileName, header=None, index_col=0, delimiter=' ',
                       quotechar=' ')


# WT is as defined in parse().
# vj is an 1xD word embedding
# Returns a nx1 vector where n is the number of embeddings.
# The delta function has been evaluated on all w_i.
def delta(WT, vj):
    # WTwj is nx1
    WTwj = np.dot(WT, vj)
    # norms is nx1
    norms = np.linalg.norm(WT, axis=1)
    # jnorm is as scalar
    jnorm = np.linalg.norm(vj)
    # denom is nx1
    denom = (norms * jnorm)**-1.0
    # entry wise multiplication of vectors
    # like dot product but without summing
    return np.multiply(WTwj, denom)


def cosadd(WT, a, b, x):
    # wx_m_wa_p_wb is 1xD
    wx_m_wa_p_wb = WT.loc[x] - WT.loc[a] + WT.loc[b]
    # d is nx1
    d = delta(WT, wx_m_wa_p_wb)
    # find the index of the maximum delta
    max_idx = np.argmax(d)
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
