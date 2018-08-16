import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from modAL.uncertainty import uncertainty_sampling

from src.utils import load_vectorize_data, transform_print, \
    objective_aware_sampling, positive_certainty_sampling, \
    random_sampling, get_init_training_data_idx
from src.active_learning import Learner, ScreeningActiveLearner

seed = 123

if __name__ == '__main__':
    predicates = ['C04', 'C12']
    file_name = 'ohsumed_C04_C12_1grams.csv'
    # load and transform data
    X, y_screening, y_predicate = load_vectorize_data(file_name, predicates, seed)

    data_df = []
    init_train_size = 50
    k = 10
    skf = StratifiedKFold(n_splits=k, random_state=seed)
    for train_idx, test_idx in skf.split(X, y_screening):
        print('-------------------------------')
        # split training-test datasets
        X_train, X_test = X[train_idx], X[test_idx]
        y_screening_train, y_screening_test = y_screening[train_idx], y_screening[test_idx]
        """create initial training dataset for all predicates
           the dataset will be further used for fine tining screening out threshold """
        init_train_idx = get_init_training_data_idx(y_screening_train, init_train_size, seed)

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
                'clf': LogisticRegression(class_weight='balanced', random_state=seed),
                'undersampling_thr': 0.333,
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
            'p_out': 0.65,
            'lr': 10,
            'learners': learners
        }
        SAL = ScreeningActiveLearner(screening_params)
        # SAL.init_stat()  # initialize statistic for predicates, uncomment if use predicate selection feature
        n_queries = 50
        beta = 3
        num_items_queried = init_train_size*len(predicates)
        data = []

        for i in range(n_queries):
            # SAL.update_stat() # uncomment if use predicate selection feature
            pr = SAL.select_predicate(i)
            query_idx, query_idx_discard = SAL.query(pr)
            SAL.teach(pr, query_idx, query_idx_discard)
            SAL.fit_meta(X_train_init, y_screening_init)

            predicted = SAL.predict(X_test)
            metrics = SAL.compute_screening_metrics(y_screening_test, predicted, SAL.lr, beta)
            pre, rec, fbeta, loss = metrics
            num_items_queried += SAL.n_instances_query
            data.append([num_items_queried, pre, rec, fbeta, loss])

            print('query no. {}: loss: {:1.3f}, fbeta: {:1.3f}, '
                          'recall: {:1.3f}, precisoin: {:1.3f}'
                  .format(i + 1, loss, fbeta, rec, pre))
            data_df.append(pd.DataFrame(data, columns=['num_items_queried',
                                                       'precision', 'recall',
                                                       'f_beta', 'loss']))

    transform_print(data_df, learner_params['sampling_strategy'].__name__,
                    predicates, 'screening_al_{}_{}'.format(predicates[0], predicates[1]))
    print('Done!')
