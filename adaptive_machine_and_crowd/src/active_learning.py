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
                'tpr': [],
                'tnr': [],
                'f_beta': [],
            }

    # def _select_predicate(self, extrapolated_val):
    #     predicate_loss = (None, float('inf'))
    #     for key, val in extrapolated_val.items():
    #         fnr = 1 - val['tpr']
    #         fpr = 1 - val['tnr']
    #         loss = self.lr * fnr + fpr
    #         if predicate_loss[1] > loss:
    #             predicate_loss = (key, loss)
    #
    #     return predicate_loss[0]

    # compute and update performance statistic for predicate-based classifiers
    def update_stat(self):
        # do cross validation
        # estimate and save statistics for extrapolation
        for predicate in self.predicates:
            s = self.stat[predicate]
            assert (len(s['num_items_queried']) == len(s['tpr']) == len(s['tnr']) == len(s['f_beta'])), 'Stat attribute error'

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
                tpr, tnr = self.compute_tpr_tnr(y_val, clf.predict(X_val))
                tpr_list.append(tpr)
                tnr_list.append(tnr)
            l.learner.fit(X, y)

            tpr_mean, tnr_mean, f_beta_mean = np.mean(tpr_list), np.mean(tnr_list), np.mean(f_beta_list)
            try:
                num_items_queried_prev = self.stat[predicate]['num_items_queried'][-1]
            except IndexError:
                num_items_queried_prev = 0
            self.stat[predicate]['num_items_queried']\
                .append((num_items_queried_prev + self.n_instances_query))
            self.stat[predicate]['tpr'].append(tpr_mean)
            self.stat[predicate]['tnr'].append(tnr_mean)
            self.stat[predicate]['f_beta'].append(f_beta_mean)

    # def extrapolate(self):
    #     extrapolated_val = {}
    #     for predicate in self.predicates:
    #         s = self.stat[predicate]
    #         num_items_queried = s['num_items_queried']
    #         f_tpr = interpolate.interp1d(num_items_queried, s['tpr'],
    #                                      fill_value='extrapolate')
    #         f_tnr = interpolate.interp1d(num_items_queried, s['tnr'],
    #                                      fill_value='extrapolate')
    #
    #         tpr = f_tpr(num_items_queried[-1] + self.n_instances_query)
    #         if tpr > 1:
    #             tpr = 1
    #         elif tpr < 0:
    #             tpr = 0
    #
    #         tnr = f_tnr(num_items_queried[-1] + self.n_instances_query)
    #         if tnr > 1:
    #             tnr = 1
    #         elif tnr < 0:
    #             tnr = 0
    #
    #         extrapolated_val[predicate] = {
    #             'tpr': tpr,
    #             'tnr': tnr
    #         }
    #
    #     return extrapolated_val


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


class ScreeningActiveLearner:

    def __init__(self, params):
        self.n_instances_query = params['n_instances_query']
        self.screening_out_threshold = params['screening_out_threshold']
        self.lr = params['lr']
        self.beta = params['beta']
        self.learners = params['learners']
        self.predicates = list(self.learners.keys())

    def select_predicate(self, param):
        if len(self.predicates) == 1:
            return self.predicates[0]
        elif len(self.predicates) == 2:
            return self.predicates[param % 2]
        else:
            raise ValueError('More than 2 predicates not supported yet. Change select_predicate method.')

    def query(self, predicate):
        l = self.learners[predicate]
        # all learners except the current one
        learners_ = {l_: self.learners[l_] for l_ in self.learners if l_ not in [predicate]}
        query_idx, _ = l.learner.query(l.X_pool,
                                       n_instances=self.n_instances_query,
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
