import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from modAL.models import ActiveLearner


class Learner:

    def __init__(self, params):
        self.clf = params['clf']
        self.n_queries = params['n_queries']
        self.n_instances_query = params['n_instances_query']
        self.undersampling_thr = params['undersampling_thr']
        self.seed = params['seed']
        self.init_train_size = params['init_train_size']
        self.sampling_strategy = params['sampling_strategy']

    def initialize_active_learner(self, X, y):
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

    def run(self, X, y, X_test, y_test):
        self.X_test, self.y_test = X_test, y_test
        self.initialize_active_learner(X, y)

        # pool-based sampling
        for idx in range(self.n_queries):
            query_idx, _ = self.learner.query(self.X_pool, n_instances=self.n_instances_query)

            # prop_of_positives = sum(learner.y_training) / len(learner.y_training)
            self.learner.teach(
                X=self.X_pool[query_idx],
                y=self.y_pool[query_idx]
            )
            # remove queried instance from pool
            self.X_pool = np.delete(self.X_pool, query_idx, axis=0)
            self.y_pool = np.delete(self.y_pool, query_idx)

            pre_, rec_, f1_, _ = precision_recall_fscore_support(self.y_test, self.learner.predict(X_test), average='binary')
            print('F1 after query no. %d: %f' % (idx + 1, f1_), end='  ')
            print('prop of + {:1.3f}'.format(sum(self.learner.y_training) / len(self.learner.y_training)))
