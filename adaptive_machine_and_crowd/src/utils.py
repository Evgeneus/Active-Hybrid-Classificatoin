import numpy as np
import pandas as pd
import warnings, random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix


class Vectorizer():
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=False, max_features=2000, ngram_range=(1, 2))

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

    def fit(self, X):
        self.vectorizer.fit(X)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X).toarray()


class CrowdSimulator:

    @staticmethod
    def crowdsource_items(item_ids, gt_items, predicate, crowd_acc, n, crowd_votes_counts):
        '''
        :param gt_items: list of ground truth values fo items to crowdsource
        :param crowd_acc: crowd accuracy range on predicate given
        :param n: n crowd votes per predicate
        :param predicate: predicate name for
        :return: aggregated crwodsourced label on items
        '''
        crodsourced_items = []
        for item_id, gt in zip(item_ids, gt_items):
            in_votes, out_votes = 0, 0
            for _ in range(n):
                worker_acc = random.uniform(crowd_acc[0], crowd_acc[1])
                worker_vote = np.random.binomial(1, worker_acc if gt == 1 else 1 - worker_acc)
                if worker_vote == 1:
                    in_votes += 1
                else:
                    out_votes += 1
            item_label = 1 if in_votes >= out_votes else 0
            crowd_votes_counts[item_id][predicate]['in'] += in_votes
            crowd_votes_counts[item_id][predicate]['out'] += out_votes
            crodsourced_items.append(item_label)
        return crodsourced_items


# screening metrics, aimed to obtain high recall
class MetricsMixin:

    @staticmethod
    def compute_screening_metrics(gt, predicted, lr, beta):
        '''
        FP == False Inclusion
        FN == False Exclusion
        '''
        item_ids = gt.keys()
        fp = 0.
        fn = 0.
        tp = 0.
        tn = 0.
        for item_id in item_ids:
            gt_val, pred_val = gt[item_id], predicted[item_id]
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

    # @staticmethod
    # def compute_tpr_tnr(gt, predicted):
    #     tn, fp, fn, tp = confusion_matrix(gt, predicted).ravel()
    #     TPR = tp / (tp + fn)  # sensitivity, recall, or true positive rate
    #     TNR = tn / (tn + fp)  # specificity or true negative rate
    #
    #     return TPR, TNR


def load_data(file_name, predicates):
    path_dict = {
        '100000_reviews_lemmatized_old.csv': '../../data/amazon-sentiment-dataset/',
        '5000_reviews_lemmatized.csv': '../../data/amazon-sentiment-dataset/',
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


def get_init_training_data_idx(y_screening, y_predicate_train, init_train_size):
   # initial training data
   pos_idx_all = (y_screening == 1).nonzero()[0]
   # all predicates are negative
   neg_idx_all = (sum(list(y_predicate_train.values())) == 0).nonzero()[0]
   # randomly select initial balanced training dataset
   train_idx = np.concatenate([np.random.choice(pos_idx_all, init_train_size // 2, replace=False),
                               np.random.choice(neg_idx_all, init_train_size // 2, replace=False)])

   return train_idx


# random sampling strategy for modAL
def random_sampling(_, X, n_instances=1):
    query_idx = np.array(random.sample(range(X.shape[0]), n_instances))

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


# sampling takes into account conjunctive expression of predicates
def mix_sampling(classifier, X, learners_, n_instances=1, **uncertainty_measure_kwargs):
    from modAL.uncertainty import classifier_uncertainty, multi_argmax
    epsilon = 0.5
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)

    if np.random.binomial(1, epsilon):
        query_idx = np.array(random.sample(range(0, X.shape[0]-1), n_instances))
    else:
        l_prob_in = np.ones(X.shape[0])
        if learners_:
            for l in learners_.values():
                l_prob_in *= l.learner.predict_proba(X)[:, 1]
            uncertainty_weighted = l_prob_in * uncertainty
        else:
            uncertainty_weighted = uncertainty

        query_idx = multi_argmax(uncertainty_weighted, n_instances=n_instances)

    return query_idx, X[query_idx]


def transform_print(data_df, file_name):
    # compute mean and std, and median over results
    columns = ['budget_spent_mean', 'precision_mean', 'recall_mean', 'f_beta_mean',
               'loss_mean', 'fn_count_mean', 'fp_count_mean']
    df_concat = pd.concat(data_df)
    policies = df_concat['policy'].unique()
    df_to_print = pd.DataFrame([], columns=columns)
    for policy in policies:
        df_policy = df_concat[df_concat['policy'] == policy]
        by_row_index = df_policy.groupby(df_policy.index)
        df_means = by_row_index.mean()
        df_std = by_row_index.std()
        df_median = by_row_index.median()

        # form dataframe for printing out in csv
        df_to_print_ = df_means
        df_to_print_.columns = columns
        df_to_print_['precision_median'] = df_median['precision']
        df_to_print_['recall_median'] = df_median['recall']
        df_to_print_['f_beta_median'] = df_median['f_beta']
        df_to_print_['loss_median'] = df_median['loss']
        df_to_print_['fn_count_median'] = df_median['fn_count']
        df_to_print_['fp_count_median'] = df_median['fp_count']
        df_to_print_['budget_spent_median'] = df_median['budget_spent']

        df_to_print_['precision_std'] = df_std['precision']
        df_to_print_['recall_std'] = df_std['recall']
        df_to_print_['f_beta_std'] = df_std['f_beta']
        df_to_print_['loss_std'] = df_std['loss']
        df_to_print_['fn_count_std'] = df_std['fn_count']
        df_to_print_['fp_count_std'] = df_std['fp_count']
        df_to_print_['budget_spent_std'] = df_std['budget_spent']
        df_to_print_['policy'] = policy
        df_to_print_['sampling_strategy'] = data_df[0]['sampling_strategy'].values[0]


        df_to_print = df_to_print.append(df_to_print_)

    df_to_print.to_csv('../output/adaptive_machines_and_crowd/{}.csv'.format(file_name), index=False)
