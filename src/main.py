import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from modAL.uncertainty import uncertainty_sampling

from src.utils import load_data
from src.active_learning import Learner

seed = 123


if __name__ == '__main__':
    # load and transform data
    df = pd.read_csv('../data/ohsumed_C14_C23_1grams.csv')
    predicate = 'C14'
    X_, y_ = load_data(predicate, df)
    vectorizer = TfidfVectorizer(lowercase=False, max_features=2000, ngram_range=(1, 1))
    X_ = vectorizer.fit_transform(X_).toarray()

    k = 10
    skf = StratifiedKFold(n_splits=k, random_state=seed)
    pre, rec, f1 = [], [], []
    for train_index, test_index in skf.split(X_, y_):
        X, X_test = X_[train_index], X_[test_index]
        y, y_test = y_[train_index], y_[test_index]

        params = {
            'clf': LogisticRegression(class_weight='balanced', random_state=seed),
            'n_queries': 500,
            'n_instances_query': 50,  # num of instances for labeling for 1 query
            'undersampling_thr': 0.33,
            'seed': seed,
            'init_train_size': 10,
            'sampling_strategy': uncertainty_sampling
        }

        Learner(params).run(X, y, X_test, y_test)
        print('-----------------')
