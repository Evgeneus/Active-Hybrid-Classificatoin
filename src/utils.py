import numpy as np


def load_data(predicate, df):
    X_pos = df.loc[df[predicate] == 1]['tokens'].values
    pos_num = len(X_pos)
    X_neg = df.loc[df[predicate] == 0]['tokens'].values
    neg_num = len(X_neg)
    X = np.append(X_pos, X_neg)
    y = np.append(np.ones(pos_num), np.zeros(neg_num))

    return X, y
