import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from machine_and_experts_annotate.src.utils import transform_print, get_init_training_data_idx, load_data, Vectorizer
from machine_and_experts_annotate.src.active_learning import Learner, ScreeningActiveLearner


def experiment_handler(experiment_params):
    file_name, n_instances_query,\
    n_queries, init_train_size, \
    screening_out_threshold, \
    test_size, beta, lr, k, \
    predicates, sampling_strategies = experiment_params
    seed = 123  # seed for Training ML algorithms

    X, y_screening, y_predicate = load_data(file_name, predicates)
    data_df = []
    # split training-test datasets
    for _ in range(k):
        train_idx, test_idx, _, _ = train_test_split(list(range(X.shape[0])), y_screening,
                                                     test_size=test_size, stratify=y_screening)
        print('-------------------------------')
        for sampling_strategy in sampling_strategies:
            print(sampling_strategy.__name__)
            vectorizer = Vectorizer()
            X_train = vectorizer.fit_transform(X[train_idx])
            X_test = vectorizer.transform(X[test_idx])

            y_screening_pool, y_screening_test = y_screening[train_idx], y_screening[test_idx]
            y_predicate_pool = {}
            for pr in predicates:
                y_predicate_pool[pr] = y_predicate[pr][train_idx]

            # creating balanced init training data
            init_train_idx = get_init_training_data_idx(y_screening_pool, y_predicate_pool, init_train_size, seed)
            y_predicate_train_init = {}
            for pr in predicates:
                y_predicate_train_init[pr] = y_predicate_pool[pr][init_train_idx]
                y_predicate_pool[pr] = np.delete(y_predicate_pool[pr], init_train_idx)
            # UNCOMMENT IF NEED TO USE
            # y_screening_init = y_screening_pool[init_train_idx]
            # y_screening_pool = np.delete(y_screening_pool, init_train_idx)

            X_train_init = X_train[init_train_idx]
            X_pool = np.delete(X_train, init_train_idx, axis=0)

            # dict of active learners per predicate
            learners = {}
            for pr in predicates:  # setup predicate-based learners
                learner_params = {
                    'clf': CalibratedClassifierCV(LinearSVC(class_weight='balanced',
                                                            C=0.1, random_state=seed)),
                    'undersampling_thr': 0.00,
                    'seed': seed,
                    'p_out': 0.5,
                    'sampling_strategy': sampling_strategy,
                }
                learner = Learner(learner_params)

                y_test = y_predicate[pr][test_idx]
                learner.setup_active_learner(X_train_init, y_predicate_train_init[pr],
                                             X_pool, y_predicate_pool[pr], X_test, y_test)
                learners[pr] = learner

            screening_params = {
                'n_instances_query': n_instances_query,  # num of instances for labeling for 1 query
                'seed': seed,
                'p_out': screening_out_threshold,
                'lr': lr,
                'beta': beta,
                'learners': learners
            }
            SAL = ScreeningActiveLearner(screening_params)
            # SAL.init_stat()  # initialize statistic for predicates, uncomment if use predicate selection feature
            num_items_queried = init_train_size*len(predicates)
            data = []

            for i in range(n_queries):
                # SAL.update_stat() # uncomment if use predicate selection feature
                pr = SAL.select_predicate(i)
                query_idx, query_idx_discard = SAL.query(pr)
                SAL.teach(pr, query_idx, query_idx_discard)
                # SAL.fit_meta(X_train_init, y_screening_init)

                predicted = SAL.predict(X_test)
                # if only one predicate -> keep y_test as predicate's y_test
                if len(predicates) == 1:
                    y_test_ = y_predicate[pr][test_idx]
                else:
                    y_test_ = y_screening_test
                metrics = SAL.compute_screening_metrics(y_test_, predicted, SAL.lr, SAL.beta)
                pre, rec, fbeta, loss, fn_count, fp_count = metrics
                num_items_queried += SAL.n_instances_query
                data.append([num_items_queried, pre, rec, fbeta, loss, fn_count, fp_count,
                             learner_params['sampling_strategy'].__name__])

                print('query no. {}: loss: {:1.3f}, fbeta: {:1.3f}, '
                      'recall: {:1.3f}, precisoin: {:1.3f}'
                      .format(i + 1, loss, fbeta, rec, pre))
                data_df.append(pd.DataFrame(data, columns=['num_items_queried',
                                                           'precision', 'recall',
                                                           'f_beta', 'loss',
                                                           'fn_count', 'fp_count',
                                                           'sampling_strategy']))

    transform_print(data_df, file_name[:-4]+'_experiment_k{}_ninstq_{}'.format(k, n_instances_query))
