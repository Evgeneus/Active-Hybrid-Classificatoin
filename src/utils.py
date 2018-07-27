import numpy as np
import pandas as pd
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer


# load and vectorize data
def load_vectorize_data(file_name, predicates, seed):
    df = pd.read_csv('../data/{}'.format(file_name))
    df_screening_pos = df.loc[df['Y'] == 1]
    df_screening_neg = df.loc[df['Y'] == 0]

    X_screening_pos = df_screening_pos['tokens'].values
    X_screening_neg = df_screening_neg['tokens'].values
    X = np.append(X_screening_pos, X_screening_neg)

    # pos_num = len(X_screening_pos)
    # neg_num = len(X_screening_neg)
    y_screening = np.append(np.ones(len(X_screening_pos)),
                            np.zeros(len(X_screening_neg)))

    y_predicate = {}  # gt labels per predicate
    for pr in predicates:
        y_predicate[pr] = np.append(df_screening_pos[pr].values,
                                    df_screening_neg[pr].values)

    # vectorize and transform text
    vectorizer = TfidfVectorizer(lowercase=False, max_features=2000, ngram_range=(1, 1))
    X = vectorizer.fit_transform(X).toarray()

    # shuffle X, y in unison
    np.random.seed(seed)
    idx = np.random.permutation(len(y_screening))
    X, y_screening = X[idx], y_screening[idx]
    for pr in predicates:
        y_predicate[pr] = y_predicate[pr][idx]

    return X, y_screening, y_predicate


# random sampling strategy for modAL
def random_sampling(_, X, n_instances=1, seed=123):
    np.random.seed(seed)
    query_idx = np.random.randint(X.shape[0], size=n_instances)

    return query_idx, X[query_idx]


# screening metrics, aimed to obtain high recall
class MetricsMixin:

    @staticmethod
    def compute_screening_metrics(gt, predicted, lr):
        '''
        FP == False Inclusion
        FN == False Exclusion
        '''
        fp = 0.
        fn = 0.
        tp = 0.
        tn = 0.
        for gt_val, pred_val in zip(gt, predicted):
            if gt_val and not pred_val:
                fn += 1
            if not gt_val and pred_val:
                fp += 1
            if gt_val and pred_val:
                tp += 1
            if not gt_val and not pred_val:
                tn += 1
        loss = (fn * lr + fp) / len(gt)
        try:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            beta = 1. / lr
            fbeta = (beta + 1) * precision * recall / (beta * recall + precision)
        except ZeroDivisionError:
            warnings.warn('ZeroDivisionError -> recall, precision, fbeta = 0., 0., 0')
            recall, precision, fbeta = 0., 0., 0

        return precision, recall, fbeta, loss
