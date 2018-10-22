import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from adaptive_machine_and_crowd.src.utils import transform_print, get_init_training_data_idx, load_data, Vectorizer
from adaptive_machine_and_crowd.src.active_learning import Learner, ScreeningActiveLearner


def run_experiment(params):
    # data_df = []
    for experiment_id in range(params['shuffling_num']):
        X, y_screening, y_predicate = load_data(params['dataset_file_name'], params['predicates'])
        vectorizer = Vectorizer()
        vectorizer.fit(X)

        params.update({
            'X': X,
            'y_screening': y_screening,
            'y_predicate': y_predicate,
            'vectorizer': vectorizer
        })

        heuristic = params['heuristic'](params)
        SAL = configure_al_box(params)
        num_items_queried = params['size_init_train_data']*len(params['predicates'])
        heuristic.update_budget_al(num_items_queried*SAL.crowd_votes_per_item)
        # data = []
        i = 0
        while heuristic.is_continue_al:
            SAL.update_stat()  # uncomment if use predicate selection feature
            pr = SAL.select_predicate(i)
            query_idx = SAL.query(pr)
            SAL.teach(pr, query_idx)

            num_items_queried += SAL.n_instances_query
            heuristic.update_budget_al(SAL.n_instances_query*SAL.crowd_votes_per_item)
            i += 1


            print('query no. {}, pr: {}, f_3 on val: {:1.2f}'.
                  format(i + 1, pr, SAL.stat[pr]['f_beta'][-1]))
            print('-------------------------')

        #     data.append([experiment_id, num_items_queried, pre, rec, fbeta, loss, fn_count, fp_count,
        #                  learner_params['sampling_strategy'].__name__] +
        #                 [SAL.stat[pred]['f_beta'][-1] for pred in predicates] +
        #                 [SAL.stat[pred]['f_beta_on_test'][-1] for pred in predicates])
        #
        #     print('query no. {}: loss: {:1.3f}, fbeta: {:1.3f}, '
        #           'recall: {:1.3f}, precisoin: {:1.3f}'
        #           .format(i + 1, loss, fbeta, rec, pre))
        # data_df.append(pd.DataFrame(data, columns=['experiment_id',
        #                                            'num_items_queried',
        #                                            'precision', 'recall',
        #                                            'f_beta', 'loss',
        #                                            'fn_count', 'fp_count',
        #                                            'sampling_strategy'] +
        #                                            ['estimated_f_beta_' + pred for pred in predicates] +
        #                                            ['f_beta_on_test_' + pred for pred in predicates]))

    # pd.concat(data_df).to_csv('../output/adaptive_machines_and_crowd/{}_adaptive_experiment_k{}_ninstq_{}_mix.csv'.
    #                           format(file_name, k, n_instances_query), index=False)
    # transform_print(data_df, file_name[:-4]+'_adaptive_experiment_k{}_ninstq_{}'.format(k, n_instances_query))


# set up active learning box
def configure_al_box(params):
    y_screening, y_predicate = params['y_screening'], params['y_predicate']
    size_init_train_data = params['size_init_train_data']
    predicates = params['predicates']

    X_pool = params['vectorizer'].transform(params['X'])
    # creating balanced init training data
    train_idx = get_init_training_data_idx(y_screening, y_predicate, size_init_train_data)
    y_predicate_train_init = {}
    X_train_init = X_pool[train_idx]
    X_pool = np.delete(X_pool, train_idx, axis=0)
    for pr in predicates:
        y_predicate_train_init[pr] = y_predicate[pr][train_idx]
        y_predicate[pr] = np.delete(y_predicate[pr], train_idx)

    # dict of active learners per predicate
    learners = {}
    for pr in predicates:  # setup predicate-based learners
        learner_params = {
            'clf': CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=0.1)),
            'sampling_strategy': params['sampling_strategy'],
        }
        learner = Learner(learner_params)
        learner.setup_active_learner(X_train_init, y_predicate_train_init[pr], X_pool, y_predicate[pr])
        learners[pr] = learner

    params.update({'learners': learners})
    SAL = ScreeningActiveLearner(params)
    SAL.init_stat()  # initialize statistic for predicates, uncomment if use predicate selection feature

    return SAL
