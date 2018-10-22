import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from adaptive_machine_and_crowd.src.utils import transform_print, get_init_training_data_idx, \
    load_data, Vectorizer, CrowdSimulator, MetricsMixin
from adaptive_machine_and_crowd.src.active_learning import Learner, ScreeningActiveLearner


def run_experiment(params):
    # parameters for crowd simulation
    crowd_acc = params['crowd_acc']
    crowd_votes_per_item = params['crowd_votes_per_item']
    predicates = params['predicates']

    results_df = []
    for experiment_id in range(params['shuffling_num']):
        print('experiment_id: ', experiment_id)

        X, y_screening, y_predicate = load_data(params['dataset_file_name'], predicates)
        vectorizer = Vectorizer()
        vectorizer.fit(X)

        items_num = y_screening.shape[0]
        item_ids_helper = {pr: np.arange(items_num) for pr in predicates}  # helper to track item ids
        item_crowd_counts = {}
        for item_id in range(items_num):
            item_crowd_counts[item_id] = {pr: {'in': 0, 'out': 0} for pr in predicates}
        item_classified = {item_id: 1 for item_id in range(items_num)}  # classify all items as in by default
        y_screening_dict = {item_id: label for item_id, label in zip(list(range(items_num)), y_screening)}

        params.update({
            'X': X,
            'y_screening': y_screening,
            'y_predicate': y_predicate,
            'vectorizer': vectorizer
        })

        heuristic = params['heuristic'](params)
        SAL = configure_al_box(params, item_ids_helper, item_crowd_counts, item_classified)
        num_items_queried = params['size_init_train_data']*len(predicates)
        heuristic.update_budget_al(num_items_queried*crowd_votes_per_item)
        results_list = []
        i = 0
        while heuristic.is_continue_al:
            print(i)
            # SAL.update_stat()  # uncomment if use predicate selection feature
            pr = SAL.select_predicate(i)
            query_idx = SAL.query(pr)

            # crowdsource sampled items
            gt_items_queried = SAL.learners[pr].y_pool[query_idx]
            y_crowdsourced = CrowdSimulator.crowdsource_items(item_ids_helper[pr][query_idx], gt_items_queried, pr,
                                                              crowd_acc[pr], crowd_votes_per_item, item_crowd_counts)
            SAL.teach(pr, query_idx, y_crowdsourced)
            item_ids_helper[pr] = np.delete(item_ids_helper[pr], query_idx)

            num_items_queried += SAL.n_instances_query
            heuristic.update_budget_al(SAL.n_instances_query*crowd_votes_per_item)
            i += 1

        # compute metrics and pint results to csv
        metrics = MetricsMixin.compute_screening_metrics(y_screening_dict, item_classified,
                                                         params['lr'], params['beta'])
        pre, rec, f_beta, loss, fn_count, fp_count = metrics
        num_items_queried += SAL.n_instances_query
        results_list.append([pre, rec, f_beta, loss, fn_count, fp_count, params['sampling_strategy'].__name__])

        print('experiment_id {}: loss: {:1.3f}, fbeta: {:1.3f}, '
              'recall: {:1.3f}, precisoin: {:1.3f}'
              .format(experiment_id, loss, f_beta, rec, pre))
        results_df.append(pd.DataFrame(results_list, columns=[
                                                   'precision', 'recall',
                                                   'f_beta', 'loss',
                                                   'fn_count', 'fp_count',
                                                   'sampling_strategy']))

    transform_print(results_df, params['dataset_file_name'][:-4] + '_shuffling_num_{}_ninstq_{}'
                    .format(params['shuffling_num'], params['n_instances_query']))


# set up active learning box
def configure_al_box(params, item_ids_helper, item_crowd_counts, item_classified):
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
        item_ids_helper[pr] = np.delete(item_ids_helper[pr], train_idx)
        for item_id, label in zip(train_idx, y_predicate_train_init[pr]):
            if label == 1:
                item_crowd_counts[item_id][pr]['in'] = params['crowd_votes_per_item']
            else:
                item_crowd_counts[item_id][pr]['out'] = params['crowd_votes_per_item']
            item_classified[item_id] = label

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
    # SAL.init_stat()  # initialize statistic for predicates, uncomment if use predicate selection feature

    return SAL
