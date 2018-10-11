import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from modAL.uncertainty import uncertainty_sampling

from machine_and_experts_annotate.src.utils import transform_print, \
    objective_aware_sampling, get_init_training_data_idx, load_data, Vectorizer
from machine_and_experts_annotate.src.active_learning import Learner, ScreeningActiveLearner

seed = 123

if __name__ == '__main__':
    predicates = ['is_negative', 'is_book']
    file_name = '100000_reviews_lemmatized.csv'
    # predicates = ['C04', 'C12']
    # file_name = 'ohsumed_C04_C12_1grams.csv'
    X, y_screening, y_predicate = load_data(file_name, predicates)

    data_df = []
    init_train_size = 20
    k = 5
    # split training-test datasets
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y_screening):
        print('-------------------------------')
        vectorizer = Vectorizer(X[train_idx])
        X_train = vectorizer.transform(X[train_idx])
        X_test = vectorizer.transform(X[test_idx])

        y_screening_train, y_screening_test = y_screening[train_idx], y_screening[test_idx]
        y_predicate_train = {}
        for pr in predicates:
            y_predicate_train[pr] = y_predicate[pr][train_idx]
        # creating balanced init training data
        init_train_idx = get_init_training_data_idx(y_screening_train, y_predicate_train, init_train_size, seed)

        y_predicate_pool, y_predicate_train_init, y_predicate_test = {}, {}, {}
        for pr in predicates:
            y_predicate_pool[pr] = y_predicate[pr][train_idx]
            y_predicate_train_init[pr] = y_predicate_pool[pr][init_train_idx]
            y_predicate_pool[pr] = np.delete(y_predicate_pool[pr], init_train_idx)

        X_train_init = X_train[init_train_idx]
        X_pool = np.delete(X_train, init_train_idx, axis=0)
        y_screening_init = y_screening_train[init_train_idx]
        y_screening_train = np.delete(y_screening_train, init_train_idx)

        # dict of active learners per predicate
        learners = {}
        for pr in predicates:  # setup predicate-based learners
            learner_params = {
                'clf': CalibratedClassifierCV(LinearSVC(class_weight='balanced',
                                                        C=0.1, random_state=seed)),
                'undersampling_thr': 0.3,
                'seed': seed,
                'p_out': 0.5,
                'sampling_strategy': objective_aware_sampling,
            }
            learner = Learner(learner_params)

            y_train_init = y_predicate[pr][train_idx][init_train_idx]
            y_pool = np.delete(y_predicate[pr][train_idx], init_train_idx)
            y_test = y_predicate[pr][test_idx]

            learner.setup_active_learner(X_train_init, y_train_init, X_pool, y_pool, X_test, y_test)
            learners[pr] = learner

        screening_params = {
            'n_instances_query': 200,  # num of instances for labeling for 1 query
            'seed': seed,
            'p_out': 0.7,
            'lr': 5,
            'beta': 3,
            'learners': learners
        }
        SAL = ScreeningActiveLearner(screening_params)
        # SAL.init_stat()  # initialize statistic for predicates, uncomment if use predicate selection feature
        n_queries = 50
        num_items_queried = init_train_size*len(predicates)
        data = []

        for i in range(n_queries):
            # SAL.update_stat() # uncomment if use predicate selection feature
            pr = SAL.select_predicate(i)
            query_idx, query_idx_discard = SAL.query(pr)
            SAL.teach(pr, query_idx, query_idx_discard)
            # SAL.fit_meta(X_train_init, y_screening_init)

            predicted = SAL.predict(X_test)
            metrics = SAL.compute_screening_metrics(y_screening_test, predicted, SAL.lr, SAL.beta)
            pre, rec, fbeta, loss, fn_count, fp_count = metrics
            num_items_queried += SAL.n_instances_query
            data.append([num_items_queried, pre, rec, fbeta, loss, fn_count, fp_count])

            print('query no. {}: loss: {:1.3f}, fbeta: {:1.3f}, '
                          'recall: {:1.3f}, precisoin: {:1.3f}'
                  .format(i + 1, loss, fbeta, rec, pre))
            data_df.append(pd.DataFrame(data, columns=['num_items_queried',
                                                       'precision', 'recall',
                                                       'f_beta', 'loss',
                                                       'fn_count', 'fp_count']))

    transform_print(data_df, learner_params['sampling_strategy'].__name__,
                    predicates, 'screening_al_{}_{}XXX'.format(predicates[0], predicates[1]))
    print('Done!')
