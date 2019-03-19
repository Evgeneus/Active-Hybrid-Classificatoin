import numpy as np
import pandas as pd
import warnings, random

from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer():
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=False, max_features=2000, ngram_range=(1, 2))

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

    def fit(self, X):
        self.vectorizer.fit(X)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X).toarray()


def crowdsource_items_al(crowd_votes, crowd_votes_counts, item_ids, predicate, n):
    crodsourced_items = []
    votes_num = 0
    for item_id in item_ids:
        in_votes, out_votes = 0, 0
        for _ in range(n):
            vote_list = crowd_votes[item_id][predicate]
            if vote_list:
                vote = vote_list.pop()
                if vote == 1:
                    in_votes += 1
                else:
                    out_votes += 1
        votes_num += in_votes + out_votes
        item_label = 1 if in_votes >= out_votes else 0
        crowd_votes_counts[item_id][predicate]['in'] += in_votes
        crowd_votes_counts[item_id][predicate]['out'] += out_votes
        crodsourced_items.append(item_label)
    return crodsourced_items, votes_num


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


def load_data(file_name, predicates):
    path_dict = {
        '100000_reviews_lemmatized_old.csv': '../../data/amazon-sentiment-dataset/',
        '5000_reviews_lemmatized.csv': '../../data/amazon-sentiment-dataset/',
        '1k_amazon_reviews_crowdsourced_lemmatized_min3votes.csv': '../../data/amazon-sentiment-dataset/',
        'ohsumed_C04_C12_1grams.csv': '../../data/ohsumed_data/',
        'ohsumed_C10_C23_1grams.csv': '../../data/ohsumed_data/',
        'ohsumed_C14_C23_1grams.csv': '../../data/ohsumed_data/',
        'loneliness-dataset-2018.csv': '../../data/loneliness-dataset-2018/',
        '1k_amazon_reviews_crowdsourced_lemmatized.csv': '../../data/amazon-sentiment-dataset/'
    }
    path = path_dict[file_name]
    data = pd.read_csv(path + file_name)
    X = data['tokens'].values
    y_screening = data['Y'].values
    y_predicate = {}  # gt labels per predicate
    for pr in predicates:
        y_predicate[pr] = data[pr].values
    crowd_votes = {}
    for item_id in range(len(y_screening)):
        crowd_votes[item_id] = {}
        for pr in predicates:
            in_num = data.loc[data['item_id'] == item_id][pr + '_in'].values[0]
            out_num = data.loc[data['item_id'] == item_id][pr + '_out'].values[0]
            votes = [1]*in_num + [0]*out_num
            random.shuffle(votes)
            crowd_votes[item_id][pr] = votes

    return X, y_screening, y_predicate, crowd_votes


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
