import numpy as np
import pandas as pd
import warnings, random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix


class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=False, max_features=2000, ngram_range=(1, 2))

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X).toarray()


# screening metrics, aimed to obtain high recall
class MetricsMixin:

    @staticmethod
    def compute_screening_metrics(gt, predicted, lr, beta):
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
            beta = beta
            fbeta = (beta ** 2 + 1) * precision * recall / (recall + beta ** 2 * precision)
        except ZeroDivisionError:
            warnings.warn('ZeroDivisionError -> recall, precision, fbeta = 0., 0., 0')
            recall, precision, fbeta = 0., 0., 0

        return precision, recall, fbeta, loss, fn, fp

    @staticmethod
    def compute_tpr_tnr(gt, predicted):
        tn, fp, fn, tp = confusion_matrix(gt, predicted).ravel()
        TPR = tp / (tp + fn)  # sensitivity, recall, or true positive rate
        TNR = tn / (tn + fp)  # specificity or true negative rate

        return TPR, TNR


def load_data(file_name, predicates):
    path_dict = {
        '100000_reviews_lemmatized.csv': '../../data/amazon-sentiment-dataset/',
        'ohsumed_C04_C12_1grams.csv': '../../data/ohsumed_data/',
        'ohsumed_C10_C23_1grams.csv': '../../data/ohsumed_data/',
        'ohsumed_C14_C23_1grams.csv': '../../data/ohsumed_data/',
        'loneliness-dataset-2018.csv': '../../data/loneliness-dataset-2018/'
    }
    path = path_dict[file_name]
    data = pd.read_csv(path + file_name)
    X = data['tokens'].values
    y_screening = data['Y'].values
    y_predicate = {}  # gt labels per predicate
    for pr in predicates:
        y_predicate[pr] = data[pr].values

    return X, y_screening, y_predicate


def get_init_training_data_idx(y_screening, y_predicate_train, init_train_size, seed):
   # initial training data
   pos_idx_all = (y_screening == 1).nonzero()[0]
   # all predicates are negative
   neg_idx_all = (sum(list(y_predicate_train.values())) == 0).nonzero()[0]
   # randomly select initial balanced training dataset
   np.random.seed(seed)
   train_idx = np.concatenate([np.random.choice(pos_idx_all, init_train_size // 2, replace=False),
                               np.random.choice(neg_idx_all, init_train_size // 2, replace=False)])

   return train_idx


# random sampling strategy for modAL
def random_sampling(_, X, n_instances=1, seed=123):
    random.seed(seed)
    query_idx = np.array(random.sample(range(X.shape[0]), n_instances))

    return query_idx, X[query_idx]


# positive class certainty sampling strategy for modAL
def positive_certainty_sampling(classifier, X, n_instances=1, **predict_proba_kwargs):
    prob_in = classifier.predict_proba(X, **predict_proba_kwargs)[:, 1]
    query_idx = np.argpartition(prob_in, -n_instances)[-n_instances:]

    return query_idx, X[query_idx]


# sampling takes into account conjunctive expression of predicates
def objective_aware_sampling(classifier, X, learners_, n_instances=1, **uncertainty_measure_kwargs):
    from modAL.uncertainty import classifier_uncertainty, multi_argmax
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    l_prob_in = np.ones(X.shape[0])
    if learners_:
        for l in learners_.values():
            l_prob_in *= l.learner.predict_proba(X)[:, 1]
        uncertainty_weighted = l_prob_in * uncertainty
    else:
        uncertainty_weighted = uncertainty

    query_idx = multi_argmax(uncertainty_weighted, n_instances=n_instances)

    return query_idx, X[query_idx]

# # sampling takes into account conjunctive expression of predicates
# def screening_sampling(classifier, X, learners_, n_instances=1, **uncertainty_measure_kwargs):
#     from modAL.uncertainty import classifier_uncertainty, multi_argmax
#     epsilon = 0.1
#     uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
#
#     # np.random.seed(seed=123)
#     if np.random.binomial(1, epsilon):
#         if learners_:
#             for l in learners_.values():
#                 uncertainty *= classifier_uncertainty(l.clf, X, **uncertainty_measure_kwargs)
#         uncertainty_weighted = uncertainty
#     else:
#         l_prob_in = np.ones(X.shape[0])
#         if learners_:
#             for l in learners_.values():
#                 l_prob_in *= l.learner.predict_proba(X)[:, 1]
#             uncertainty_weighted = l_prob_in * uncertainty
#         else:
#             uncertainty_weighted = uncertainty
#
#     query_idx = multi_argmax(uncertainty_weighted, n_instances=n_instances)
#
#     return query_idx, X[query_idx]


# transfrom data from k-fold CV and print results in csv
def transform_print(data_df, sampl_strategy, predicates, file_name):
    # compute mean and std, and median over k-fold cross validation results
    df_concat = pd.concat(data_df)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_std = by_row_index.std()
    df_median = by_row_index.median()

    # form dataframe for printing out in csv
    df_to_print = df_means
    df_to_print.columns = ['num_items_queried', 'precision_mean',
                           'recall_mean', 'f_beta_mean', 'loss_mean',
                           'fn_count_mean', 'fp_count_mean']

    df_to_print['precision_median'] = df_median['precision']
    df_to_print['recall_median'] = df_median['recall']
    df_to_print['f_beta_median'] = df_median['f_beta']
    df_to_print['loss_median'] = df_median['loss']
    df_to_print['fn_count_median'] = df_median['fn_count']
    df_to_print['fp_count_median'] = df_median['fp_count']

    df_to_print['precision_std'] = df_std['precision']
    df_to_print['recall_std'] = df_std['recall']
    df_to_print['f_beta_std'] = df_std['f_beta']
    df_to_print['loss_std'] = df_std['loss']
    df_to_print['fn_count_std'] = df_std['fn_count']
    df_to_print['fp_count_std'] = df_std['fp_count']


    df_to_print['sampling_strategy'] = sampl_strategy
    df_to_print.to_csv('../output/machines_and_experts/{}.csv'.format(file_name), index=False)
