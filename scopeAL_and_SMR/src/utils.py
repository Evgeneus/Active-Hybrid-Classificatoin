import numpy as np
import pandas as pd
import warnings, random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score


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


def load_data(file_name, predicates, path_to_project):
    path_dict = {
        '100000_reviews_lemmatized_old.csv': path_to_project + 'data/amazon-sentiment-dataset/',
        '5000_reviews_lemmatized.csv': path_to_project + 'data/amazon-sentiment-dataset/',
        'ohsumed_C04_C12_1grams.csv': path_to_project + 'data/ohsumed_data/',
        'ohsumed_C10_C23_1grams.csv': path_to_project + '/data/ohsumed_data/',
        'ohsumed_C14_C23_1grams.csv': path_to_project + 'data/ohsumed_data/',
        'loneliness-dataset-2018.csv': path_to_project + 'data/loneliness-dataset-2018/'
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
    query_idx = random.sample(range(X.shape[0]), n_instances)

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


# Mixin for ScreeningActiveLearner if to use adaptive_policy for learning-exploitation
class ChoosePredicateMixin:

    def init_stat(self):
        # initialize statistic for predicates
        self.stat = {}
        for predicate in self.predicates:
            self.stat[predicate] = {
                'num_items_queried': [],
                'f_beta': [],
            }

    # compute and update performance statistic for predicate-based classifiers
    def update_stat(self):
        # do cross validation
        # estimate and save statistics for extrapolation
        window = 5
        for predicate in self.predicates:
            s = self.stat[predicate]
            assert (len(s['num_items_queried']) == len(s['f_beta'])), 'Stat attribute error'

            l = self.learners[predicate]
            X, y = l.learner.X_training, l.learner.y_training
            tpr_list, tnr_list, f_beta_list = [], [], []
            k = 5
            skf = StratifiedKFold(n_splits=k)
            for train_idx, val_idx in skf.split(np.empty(y.shape[0]), y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                clf = l.learner
                clf.fit(X_train, y_train)
                f_beta_list.append(fbeta_score(y_val, clf.predict(X_val), beta=self.beta, average='binary'))
            l.learner.fit(X, y)

            f_beta_mean = np.mean(f_beta_list)
            try:
                num_items_queried_prev = self.stat[predicate]['num_items_queried'][-1]
            except IndexError:
                num_items_queried_prev = 0

            if len(self.stat[predicate]['num_items_queried']) >= window - 1:
                f_beta_avg = (sum(self.stat[predicate]['f_beta'][-(window-1):]) + f_beta_mean) / window
                self.stat[predicate]['f_beta'].append(f_beta_avg)
            else:
                self.stat[predicate]['f_beta'].append(f_beta_mean)
            self.stat[predicate]['num_items_queried'].append((num_items_queried_prev + self.n_instances_query))

    def select_predicate_stop(self, param):
        predicates_to_train = []
        for predicate in self.predicates:
            if (self.stat[predicate]['f_beta'][-1] - self.stat[predicate]['f_beta'][-10]) >= 0.02:
                predicates_to_train.append(predicate)
        if not predicates_to_train:
            return None
        else:
            n = len(predicates_to_train)
            return predicates_to_train[param % n]
