import numpy as np
import pandas as pd
from modAL.models import ActiveLearner
from src.utils import MetricsMixin


class Learner(MetricsMixin):

    def __init__(self, params):
        self.clf = params['clf']
        self.n_queries = params['n_queries']
        self.n_instances_query = params['n_instances_query']
        self.undersampling_thr = params['undersampling_thr']
        self.seed = params['seed']
        self.init_train_size = params['init_train_size']
        self.sampling_strategy = params['sampling_strategy']
        self.p_out = params['p_out']
        self.lr = params['lr']

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

    def undersample(self, query_idx):
        pos_y_num = sum(self.learner.y_training)
        train_y_num = len(self.learner.y_training)

        pos_y_idx = (self.y_pool[query_idx] == 1).nonzero()[0]  # add all positive items from queried items
        query_idx_new = list(query_idx[pos_y_idx])                         # delete positive idx from queried query_idx
        query_neg_idx = np.delete(query_idx, pos_y_idx)

        pos_y_num += len(pos_y_idx)
        train_y_num += len(pos_y_idx)
        for y_neg_idx in query_neg_idx:
            # compute current proportion of positive items in training dataset
            if pos_y_num / train_y_num > self.undersampling_thr:
                query_idx_new.append(y_neg_idx)
                train_y_num += 1
            else:
                return query_idx_new

        return query_idx_new

    def run(self, X, y, X_test, y_test):
        self.X_test, self.y_test = X_test, y_test
        self.initialize_active_learner(X, y)

        # estimate initial metrics
        pre_, rec_, fbeta_, loss_ = self.compute_screening_metrics(self.y_test,
                                                                   self.learner.predict_proba(X_test),
                                                                   self.p_out, self.lr)
        num_items_queried = self.init_train_size
        proportion_positives = sum(self.learner.y_training) / len(self.learner.y_training)
        data = [[num_items_queried, self.init_train_size, proportion_positives,
                 pre_, rec_, fbeta_, loss_]]  # [num_items_queried, training_size, precision, recall, f1]
        # pool-based sampling
        for idx in range(self.n_queries):
            query_idx, _ = self.learner.query(self.X_pool, n_instances=self.n_instances_query)
            num_items_queried += self.n_instances_query
            query_idx_new = self.undersample(query_idx)   # undersample the majority class

            self.learner.teach(
                X=self.X_pool[query_idx_new],
                y=self.y_pool[query_idx_new]
            )
            # remove queried instance from pool
            self.X_pool = np.delete(self.X_pool, query_idx, axis=0)
            self.y_pool = np.delete(self.y_pool, query_idx)

            pre_, rec_, fbeta_, loss_ = self.compute_screening_metrics(self.y_test,
                                                                       self.learner.predict_proba(X_test),
                                                                       self.p_out, self.lr)
            proportion_positives = sum(self.learner.y_training) / len(self.learner.y_training)
            data.append([num_items_queried, len(self.learner.y_training),
                         proportion_positives, pre_, rec_, fbeta_, loss_])

            print('query no. {}: loss: {:1.3f}, fbeta: {:1.3f},'
                  'recall: {:1.3f}, precisoin: {:1.3f}'
                  .format(idx + 1, loss_, fbeta_, rec_, pre_), end='  ')
            print('prop of + {:1.3f}'.format(proportion_positives))

        print('-----------------')
        return pd.DataFrame(data, columns=['num_items_queried',
                                           'training_size',
                                           'proportion_positives',
                                           'precision',
                                           'recall',
                                           'fbeta', 'loss'])
