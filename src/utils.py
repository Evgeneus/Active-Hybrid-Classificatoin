import numpy as np
import warnings, random


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
def random_sampling(_, X, n_instances=1):
    query_idx = random.sample(range(X.shape[0]), n_instances)

    return query_idx, X[query_idx]


# screening metrics, aimed to obtain high recall
class MetricsMixin:

    @staticmethod
    def compute_screening_metrics(gt, predicted_p, p_out, lr):
        '''
        FP == False Inclusion
        FN == False Exclusion
        '''
        predicted = [0 if pred_out > p_out else 1 for pred_out in predicted_p[:, 0]]
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
