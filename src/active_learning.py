import numpy as np
from scipy import interpolate
from modAL.models import ActiveLearner

from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from sklearn.metrics import fbeta_score, make_scorer

from .utils import MetricsMixin


class ActiveLearner(ActiveLearner):

    def query(self, X, learners_=None, **query_kwargs):
        if self.query_strategy.__name__ not in ['objective_aware_sampling']:
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
                'tnr': []
            }

    def select_predicate(self, param):
        self.update_stat()

        num_items_queried_all = (param + 1) * self.n_instances_query
        if num_items_queried_all / len(self.predicates) < 100:
            return self.predicates[param % 2]

        extrapolated_val = self.extrapolate()
        predicate = self._select_predicate(extrapolated_val)

        return predicate

    def _select_predicate(self, extrapolated_val):
        predicate_loss = (None, float('inf'))
        for key, val in extrapolated_val.items():
            fnr = 1 - val['tpr']
            fpr = 1 - val['tnr']
            loss = self.lr * fnr + fpr
            if predicate_loss[1] > loss:
                predicate_loss = (key, loss)

        return predicate_loss[0]

    # compute and update performance statistic for predicate-based classifiers
    def update_stat(self):
        # do cross validation
        # estimate and save statistics for extrapolation
        for predicate in self.predicates:
            s = self.stat[predicate]
            assert (len(s['num_items_queried']) == len(s['tpr']) == len(s['tnr'])), 'Stat attribute error'

            l = self.learners[predicate]
            X, y = l.learner.X_training, l.learner.y_training

            tpr_list, tnr_list = [], []
            k = 5
            skf = StratifiedKFold(n_splits=k, random_state=self.seed)
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                clf = l.learner.fit(X_train, y_train)
                tpr, tnr = self.compute_tpr_tnr(y_val, clf.predict(X_val))
                tpr_list.append(tpr)
                tnr_list.append(tnr)
            l.learner.fit(X, y)

            tpr_mean, tnr_mean = np.mean(tpr_list), np.mean(tnr_list)
            try:
                num_items_queried_prev = self.stat[predicate]['num_items_queried'][-1]
            except IndexError:
                num_items_queried_prev = 0
            self.stat[predicate]['num_items_queried']\
                .append(num_items_queried_prev + self.n_instances_query)
            self.stat[predicate]['tpr'].append(tpr_mean)
            self.stat[predicate]['tnr'].append(tnr_mean)

    def extrapolate(self):
        extrapolated_val = {}
        for predicate in self.predicates:
            s = self.stat[predicate]
            num_items_queried = s['num_items_queried']
            f_tpr = interpolate.interp1d(num_items_queried, s['tpr'],
                                         fill_value='extrapolate')
            f_tnr = interpolate.interp1d(num_items_queried, s['tnr'],
                                         fill_value='extrapolate')

            tpr = f_tpr(num_items_queried[-1] + self.n_instances_query)
            if tpr > 1:
                tpr = 1
            elif tpr < 0:
                tpr = 0

            tnr = f_tnr(num_items_queried[-1] + self.n_instances_query)
            if tnr > 1:
                tnr = 1
            elif tnr < 0:
                tnr = 0

            extrapolated_val[predicate] = {
                'tpr': tpr,
                'tnr': tnr
            }

        return extrapolated_val


class Learner(MetricsMixin):

    def __init__(self, params):
        self.clf = params['clf']
        self.undersampling_thr = params['undersampling_thr']
        self.seed = params['seed']
        self.init_train_size = params['init_train_size']
        self.sampling_strategy = params['sampling_strategy']
        self.p_out = 0.5

    def setup_active_learner(self, X, y, X_test, y_test):
        self.X_test, self.y_test = X_test, y_test

        # initial training data
        pos_idx_all = (y == 1).nonzero()[0]
        neg_idx_all = (y == 0).nonzero()[0]
        # randomly select initial balanced training dataset
        np.random.seed(self.seed)
        train_idx = np.concatenate([np.random.choice(pos_idx_all, self.init_train_size // 2, replace=False),
                                    np.random.choice(neg_idx_all, self.init_train_size // 2, replace=False)])
        X_train = X[train_idx]
        y_train = y[train_idx]

        # generate the pool
        self.X_pool = np.delete(X, train_idx, axis=0)
        self.y_pool = np.delete(y, train_idx)

        # initialize active learner
        self.learner = ActiveLearner(
            estimator=self.clf,
            X_training=X_train, y_training=y_train,
            query_strategy=self.sampling_strategy
        )

    def undersample(self, query_idx):
        pos_y_num = sum(self.learner.y_training)
        train_y_num = len(self.learner.y_training)

        pos_y_idx = (self.y_pool[query_idx] == 1).nonzero()[0]  # add all positive items from queried items
        query_idx_new = list(query_idx[pos_y_idx])                        # delete positive idx from queried query_idx
        query_idx_discard = []
        query_neg_idx = np.delete(query_idx, pos_y_idx)

        pos_y_num += len(pos_y_idx)
        train_y_num += len(pos_y_idx)
        for y_neg_idx in query_neg_idx:
            # compute current proportion of positive items in training dataset
            if pos_y_num / train_y_num > self.undersampling_thr:
                query_idx_new.append(y_neg_idx)
                train_y_num += 1
            else:
                # return query_idx_new
                query_idx_discard.append(y_neg_idx)

        return query_idx_new, query_idx_discard


# class ScreeningActiveLearner(MetricsMixin, ChoosePredicateMixin):
class ScreeningActiveLearner(MetricsMixin):

    def __init__(self, params):
        self.n_instances_query = params['n_instances_query']
        self.seed = params['seed']
        self.p_out = params['p_out']
        self.lr = params['lr']
        self.learners = params['learners']
        self.predicates = list(self.learners.keys())

    def select_predicate(self, param):
        return self.predicates[param % 2]

    def query(self, predicate):
        l = self.learners[predicate]
        # all learners except the current one
        learners_ = {l_: self.learners[l_] for l_ in self.learners if l_ not in [predicate]}
        query_idx, _ = l.learner.query(l.X_pool,
                                       n_instances=self.n_instances_query,
                                       learners_=learners_
                                       )
        query_idx_new, query_idx_discard = l.undersample(query_idx)       # undersample the majority class

        return query_idx_new, query_idx_discard

    def teach(self, predicate, query_idx, query_idx_discard):
        l = self.learners[predicate]
        X = np.concatenate((l.learner.X_training, l.X_pool[query_idx]))
        y = np.concatenate((l.learner.y_training, l.y_pool[query_idx]))
        # remove queried instance from pool
        l.X_pool = np.delete(l.X_pool, np.concatenate((query_idx, query_idx_discard)), axis=0)
        l.y_pool = np.delete(l.y_pool, np.concatenate((query_idx, query_idx_discard)))

        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
        }
        k = 5
        grid = GridSearchCV(l.learner.estimator, cv=k, param_grid=param_grid,
                            scoring='f1', refit=True, n_jobs=-1)
        grid.fit(X, y)
        l.learner.estimator = grid.best_estimator_
        l.learner.fit(X, y)

    def predict_proba(self, X):
        proba_in = np.ones(X.shape[0])
        for l in self.learners.values():
            proba_in *= l.learner.predict_proba(X)[:, 1]
        proba = np.stack((1-proba_in, proba_in), axis=1)

        return proba

    def predict(self, X):
        proba_out = self.predict_proba(X)[:, 0]
        predicted = [0 if p > self.p_out else 1 for p in proba_out]

        return predicted
