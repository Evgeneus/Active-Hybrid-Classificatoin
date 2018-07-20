import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, \
                                            TfidfTransformer, \
                                            TfidfVectorizer
from modAL.models import ActiveLearner
from sklearn.linear_model import LogisticRegression
from modAL.uncertainty import uncertainty_sampling

seed = 123


# class LinearSVC_proba(LinearSVC):
#
#     def __platt_func(self,x):
#         return 1/(1+np.exp(-x))
#
#     def predict_proba(self, X):
#         f = np.vectorize(self.__platt_func)
#         raw_predictions = self.decision_function(X)
#         platt_predictions = f(raw_predictions)
#         probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
#         return probs

def get_data(predicate, df):
    X_pos = df.loc[df[predicate] == 1]['tokens'].values
    pos_num = len(X_pos)
    X_neg = df.loc[df[predicate] == 0]['tokens'].values
    neg_num = len(X_neg)
    X = np.append(X_pos, X_neg)
    y = np.append(np.ones(pos_num), np.zeros(neg_num))

    return X, y


def do_active_learning(X, y, X_test, y_test, n_queries, n_instances):
    # initial training data
    pos_idx_all = (y == 1).nonzero()[0]
    neg_idx_all = (y == 0).nonzero()[0]

    # randomly select initial balanced training dataset
    init_train_size = 10
    np.random.seed(seed)
    train_idx = np.concatenate([np.random.choice(pos_idx_all, init_train_size // 2, replace=False),
                                np.random.choice(neg_idx_all, init_train_size // 2, replace=False)])
    X_train = X[train_idx]
    y_train = y[train_idx]

    # generating the pool
    X_pool = np.delete(X, train_idx, axis=0)
    y_pool = np.delete(y, train_idx)

    # initializing the active learner
    learner = ActiveLearner(
        estimator=LogisticRegression(class_weight='balanced', random_state=seed),
        X_training=X_train, y_training=y_train,
        query_strategy=uncertainty_sampling
    )

    # pool-based sampling
    for idx in range(n_queries):
        query_idx, _ = learner.query(X_pool, n_instances=n_instances)
        learner.teach(
            X=X_pool[query_idx],
            y=y_pool[query_idx]
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

        pre_, rec_, f1_, _ = precision_recall_fscore_support(y_test, learner.predict(X_test), average='binary')
        print('F1 after query no. %d: %f' % (idx + 1, f1_), end='  ')
        print('prop of + {:1.3f}'.format(sum(learner.y_training)/len(learner.y_training)))


if __name__ == '__main__':
    df = pd.read_csv('./data/ohsumed_C14_C23_1grams.csv')
    predicate = 'C14'
    n_queries = 5000
    n_instances = 50  # num of instances for labeling for 1 query

    # load data
    X_, y_ = get_data(predicate, df)
    vectorizer = TfidfVectorizer(lowercase=False, max_features=1000, ngram_range=(1, 1))
    X_ = vectorizer.fit_transform(X_).toarray()

    k = 10
    skf = StratifiedKFold(n_splits=k, random_state=seed)
    pre, rec, f1 = [], [], []
    for train_index, test_index in skf.split(X_, y_):
        X, X_test = X_[train_index], X_[test_index]
        y, y_test = y_[train_index], y_[test_index]

        do_active_learning(X, y, X_test, y_test, n_queries, n_instances)
        print('-----------------')
