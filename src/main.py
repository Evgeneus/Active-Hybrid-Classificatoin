import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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

    data_df = []
    k = 10
    skf = StratifiedKFold(n_splits=k, random_state=seed)
    for train_index, test_index in skf.split(X_, y_):
        X, X_test = X_[train_index], X_[test_index]
        y, y_test = y_[train_index], y_[test_index]

        params = {
            'clf': CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=seed)),
            'n_queries': 50,
            'n_instances_query': 50,   # num of instances for labeling for 1 query
            'undersampling_thr': 0.333,
            'seed': seed,
            'init_train_size': 10,
            'sampling_strategy': uncertainty_sampling,
            'p_out': 0.5,
            'lr': 10
        }

        # start active learning
        df_run = Learner(params).run(X, y, X_test, y_test)
        data_df.append(df_run)

    # compute mean and std, and median over k-fold cross validation results
    df_concat = pd.concat(data_df)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_std = by_row_index.std()
    df_median = by_row_index.median()

    # form dataframe for printing out in csv
    df_to_print = df_means
    df_to_print.columns = ['num_items_queried', 'training_size_mean',
                           'proportion_positives_mean', 'precision_mean',
                           'recall_mean', 'fbeta_mean', 'loss_mean']

    df_to_print['training_size_median'] = df_median['training_size']
    df_to_print['precision_median'] = df_median['precision']
    df_to_print['recall_median'] = df_median['recall']
    df_to_print['fbeta_median'] = df_median['fbeta']
    df_to_print['loss_median'] = df_median['loss']

    df_to_print['training_size_std'] = df_std['training_size']
    df_to_print['precision_std'] = df_std['precision']
    df_to_print['recall_std'] = df_std['recall']
    df_to_print['fbeta_std'] = df_std['fbeta']
    df_to_print['loss_std'] = df_std['loss']

    df_to_print['sampling_strategy'] = params['sampling_strategy'].__name__
    df_to_print.to_csv('../data/single_classifier_al/screening_al_{}.csv'.format(predicate), index=False)
