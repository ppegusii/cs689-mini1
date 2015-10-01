import pandas as pd


# Returns the file parsed to a pandas dataframe WT, in which row_i is the
# vector embedding of word_i and is also indexed by word_i.
def embeddings(fileName):
    # Note quotechar is the same as the delimiter.
    # This is required because read_csv defaults to using '"' as the
    # quotechar, which causes an EOF error because '"' is a word.
    # Work around given https://github.com/pydata/pandas/issues/5500
    return pd.read_csv(fileName, header=None, index_col=0, delimiter=' ',
                       quotechar=' ')
