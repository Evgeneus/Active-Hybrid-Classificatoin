import operator
import numpy as np
from scipy import interpolate
from modAL.models import ActiveLearner
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import fbeta_score
from .utils import MetricsMixin, CrowdSimulator


class ActiveLearner(ActiveLearner):

    def query(self, X, learners_=None, **query_kwargs):
        if self.query_strategy.__name__ not in ['mix_sampling', 'objective_aware_sampling']:
            query_idx, query_instances = self.query_strategy(self, X, **query_kwargs)
        else:
            query_idx, query_instances = self.query_strategy(self, X, learners_, **query_kwargs)

        return query_idx, query_instances


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
            y = np.array(y)
            tpr_list, tnr_list, f_beta_list = [], [], []
            k = 5
            skf = StratifiedKFold(n_splits=k)
            for train_idx, val_idx in skf.split(np.empty(len(y)), y):
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


class Learner:

    def __init__(self, params):
        self.clf = params['clf']
        self.sampling_strategy = params['sampling_strategy']
        self.screening_out_threshold = params.get('screening_out_threshold', 0.5)

    def setup_active_learner(self, X_train_init, y_train_init, X_pool, y_pool):
        # generate the pool
        self.X_pool = X_pool
        self.y_pool = y_pool

        # initialize active learner
        self.learner = ActiveLearner(
            estimator=self.clf,
            X_training=X_train_init, y_training=y_train_init,
            query_strategy=self.sampling_strategy
        )


class ScreeningActiveLearner(ChoosePredicateMixin):

    def __init__(self, params):
        self.n_instances_query = params['n_instances_query']
        self.screening_out_threshold = params['screening_out_threshold']
        self.lr = params['lr']
        self.beta = params['beta']
        self.learners = params['learners']
        self.predicates = list(self.learners.keys())
        self.predicate_queue = list(range(len(self.predicates)))

    def select_predicate(self):
        pred_id = self.predicate_queue.pop(0)
        self.predicate_queue.append(pred_id)

        return self.predicates[pred_id]

    def query(self, predicate):
        l = self.learners[predicate]
        # all learners except the current one
        learners_ = {l_: self.learners[l_] for l_ in self.learners if l_ not in [predicate]}
        if self.n_instances_query > len(l.y_pool):
            if len(l.y_pool) == 0:
                return []
            n_instances = len(l.y_pool)
        else:
            n_instances = self.n_instances_query

        query_idx, _ = l.learner.query(l.X_pool,
                                       n_instances=n_instances,
                                       learners_=learners_)
        return query_idx

    def teach(self, predicate, query_idx, y_crowdsourced):
        l = self.learners[predicate]
        l.learner.teach(l.X_pool[query_idx], y_crowdsourced)
        # remove queried instance from pool
        l.X_pool = np.delete(l.X_pool, query_idx, axis=0)
        l.y_pool = np.delete(l.y_pool, query_idx)

    def predict_proba(self, X):
        proba_in = np.ones(X.shape[0])
        for l in self.learners.values():
            proba_in *= l.learner.predict_proba(X)[:, 1]
        proba = np.stack((1-proba_in, proba_in), axis=1)

        return np.array(proba)

    def predict(self, X):
        proba_out = self.predict_proba(X)[:, 0]
        predicted = [0 if p > self.screening_out_threshold else 1 for p in proba_out]

        return np.array(predicted)
