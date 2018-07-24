import numpy as np

# load data from csv
def load_data(predicate, df):
    X_pos = df.loc[df[predicate] == 1]['tokens'].values
    pos_num = len(X_pos)
    X_neg = df.loc[df[predicate] == 0]['tokens'].values
    neg_num = len(X_neg)
    X = np.append(X_pos, X_neg)
    y = np.append(np.ones(pos_num), np.zeros(neg_num))

    return X, y


# random sampling strategy for modAL
def random_sampling(_, X, n_instances=1, seed=123):
    np.random.seed(seed)
    query_idx = np.random.randint(X.shape[0], size=n_instances)
    return query_idx, X[query_idx]
