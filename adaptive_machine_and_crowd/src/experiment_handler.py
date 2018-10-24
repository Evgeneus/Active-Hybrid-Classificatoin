import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from adaptive_machine_and_crowd.src.utils import transform_print, get_init_training_data_idx, \
    load_data, Vectorizer, CrowdSimulator, MetricsMixin
from adaptive_machine_and_crowd.src.active_learning import Learner, ScreeningActiveLearner
from adaptive_machine_and_crowd.src.sm_run.shortest_multi_run import ShortestMultiRun


def run_experiment(params):
    # parameters for crowd simulation
    crowd_acc = params['crowd_acc']
    crowd_votes_per_item = params['crowd_votes_per_item']
    predicates = params['predicates']

    results_df = []
    for policy in params['policies']:
        print("Policy: {}".format(policy.name))
        print('************************************')
        for experiment_id in range(params['shuffling_num']):
            policy.B_al_spent,  policy.B_crowd_spent = 0, 0

            X, y_screening, y_predicate = load_data(params['dataset_file_name'], predicates)
            vectorizer = Vectorizer()
            vectorizer.fit(X)

            items_num = y_screening.shape[0]
            item_predicate_gt = {}
            for pr in predicates:
                item_predicate_gt[pr] = {item_id: gt_val for item_id, gt_val in zip(list(range(items_num)), y_predicate[pr])}
            item_ids_helper = {pr: np.arange(items_num) for pr in predicates}  # helper to track item ids
            crowd_votes_counts = {}
            for item_id in range(items_num):
                crowd_votes_counts[item_id] = {pr: {'in': 0, 'out': 0} for pr in predicates}
            item_labels = {item_id: 1 for item_id in range(items_num)}  # classify all items as in by default
            y_screening_dict = {item_id: label for item_id, label in zip(list(range(items_num)), y_screening)}

            params.update({
                'X': X,
                'y_screening': y_screening,
                'y_predicate': y_predicate,
                'vectorizer': vectorizer
            })

            SAL = configure_al_box(params, item_ids_helper, crowd_votes_counts, item_labels)
            policy.update_budget_al(params['size_init_train_data']*len(predicates)*crowd_votes_per_item)
            results_list = []
            i = 0
            while policy.is_continue_al:
                # SAL.update_stat()  # uncomment if use predicate selection feature
                pr = SAL.select_predicate(i)
                query_idx = SAL.query(pr)

                # crowdsource sampled items
                gt_items_queried = SAL.learners[pr].y_pool[query_idx]
                y_crowdsourced = CrowdSimulator.crowdsource_items(item_ids_helper[pr][query_idx], gt_items_queried, pr,
                                                                  crowd_acc[pr], crowd_votes_per_item, crowd_votes_counts)
                SAL.teach(pr, query_idx, y_crowdsourced)
                item_ids_helper[pr] = np.delete(item_ids_helper[pr], query_idx)

                policy.update_budget_al(SAL.n_instances_query*crowd_votes_per_item)
                i += 1
            print('experiment_id {}, AL-Box finished'.format(experiment_id), end=', ')

            # DO SM-RUN
            smr_params = {
                'estimated_predicate_accuracy': {
                    predicates[0]: 0.9,
                    predicates[1]: 0.9
                },
                'estimated_predicate_selectivity': {
                    predicates[0]: 0.30,
                    predicates[1]: 0.50
                },
                'predicates': predicates,
                'item_predicate_gt': item_predicate_gt,
                'clf_threshold': 0.9,
                'stop_score': 300,
                'crowd_acc': crowd_acc
            }
            SMR = ShortestMultiRun(smr_params)
            unclassified_item_ids = SMR.classify_items(np.arange(items_num), crowd_votes_counts, item_labels)
            while policy.is_continue_crowd and unclassified_item_ids.any():
                unclassified_item_ids, budget_round = SMR.do_round(crowd_votes_counts, unclassified_item_ids, item_labels)
                policy.update_budget_crowd(budget_round)
            print('Crowd-Box finished')

            # compute metrics and pint results to csv
            metrics = MetricsMixin.compute_screening_metrics(y_screening_dict, item_labels,
                                                             params['lr'], params['beta'])
            pre, rec, f_beta, loss, fn_count, fp_count = metrics
            budget_spent = policy.B_al_spent + policy.B_crowd_spent
            results_list.append([budget_spent, pre, rec, f_beta, loss, fn_count,
                                 fp_count, params['sampling_strategy'].__name__,
                                 policy.name])

            print('loss: {:1.3f}, fbeta: {:1.3f}, '
                  'recall: {:1.3f}, precisoin: {:1.3f}'
                  .format(loss, f_beta, rec, pre))
            print('--------------------------------------------------------------')
            results_df.append(pd.DataFrame(results_list, columns=['budget_spent',
                                                       'precision', 'recall',
                                                       'f_beta', 'loss',
                                                       'fn_count', 'fp_count',
                                                       'sampling_strategy',
                                                       'policy']))

    transform_print(results_df, params['dataset_file_name'][:-4] + '_shuffling_num_{}_ninstq_{}'
                    .format(params['shuffling_num'], params['n_instances_query']))


# set up active learning box
def configure_al_box(params, item_ids_helper, crowd_votes_counts, item_labels):
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
                crowd_votes_counts[item_id][pr]['in'] = params['crowd_votes_per_item']
            else:
                crowd_votes_counts[item_id][pr]['out'] = params['crowd_votes_per_item']
            item_labels[item_id] = label

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
