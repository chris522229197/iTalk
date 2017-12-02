import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Find the set of unique characters (vocabulary) for a particular column in a data frame
def find_vocab(df, col):
    col_str = df[col].astype(str)
    vocab = set()
    for token in col_str:
        vocab.update(list(token))
    return vocab

# Create index reference for a set of characters
def create_idx(char_set):
    char_list = sorted(char_set)
    idx_lookup = dict([(char, i) for i, char in enumerate(char_list)])
    char_lookup = dict([(i, char) for char, i in idx_lookup.items()])
    return idx_lookup, char_lookup

# Find the occurance of vocabulary character in each token of a data frame
# Return an np array with the same number of rows as the input data frame and with the same
# number of columns as the vocabulary size
def count_char(df, col, idx_lookup):
    cv = CountVectorizer(analyzer='char', lowercase=False, vocabulary=idx_lookup)
    return cv.fit_transform(df[col].astype(str))