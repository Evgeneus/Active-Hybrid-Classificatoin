import numpy as np
from modAL.models import ActiveLearner


class ActiveLearner(ActiveLearner):

    def query(self, X, learners_=None, **query_kwargs):
        if self.query_strategy.__name__ not in ['mix_sampling', 'objective_aware_sampling']:
            query_idx, query_instances = self.query_strategy(self, X, **query_kwargs)
        else:
            query_idx, query_instances = self.query_strategy(self, X, learners_, **query_kwargs)

        return query_idx, query_instances


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
                return 0
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
