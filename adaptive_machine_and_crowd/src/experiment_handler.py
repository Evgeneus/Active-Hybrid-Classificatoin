import os
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from adaptive_machine_and_crowd.src.utils import get_init_training_data_idx, \
    load_data, Vectorizer, CrowdSimulator, MetricsMixin
from adaptive_machine_and_crowd.src.active_learning import Learner, ScreeningActiveLearner
from adaptive_machine_and_crowd.src.sm_run.shortest_multi_run import ShortestMultiRun
from adaptive_machine_and_crowd.src.policy import PointSwitchPolicy


def run_experiment(params):
    # parameters for crowd simulation
    crowd_acc = params['crowd_acc']
    crowd_votes_per_item_al = params['crowd_votes_per_item_al']
    predicates = params['predicates']
    screening_out_threshold_machines = 0.7

    df_to_print = pd.DataFrame()
    for budget_per_item in params['budget_per_item']:
        B = params['dataset_size'] * budget_per_item
        for switch_point in params['policy_switch_point']:
            print('Policy switch point: {}'.format(switch_point))
            print('Budget per item: {}'.format(budget_per_item))
            print('************************************')
            results_list = []
            for experiment_id in range(params['experiment_nums']):
                policy = PointSwitchPolicy(B, switch_point)

                X, y_screening, y_predicate = load_data(params['dataset_file_name'], predicates)
                vectorizer = Vectorizer()
                vectorizer.fit(X)

                items_num = y_screening.shape[0]
                item_predicate_gt = {}
                for pr in predicates:
                    item_predicate_gt[pr] = {item_id: gt_val for item_id, gt_val in zip(list(range(items_num)), y_predicate[pr])}
                item_ids_helper = {pr: np.arange(items_num) for pr in predicates}  # helper to track item ids
                crowd_votes_counts, prior_prob = {}, {}
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

                # if Available Budget for Active Learniong is available then Do Run Active Learning Box
                if switch_point != 0:
                    SAL = configure_al_box(params, item_ids_helper, crowd_votes_counts, item_labels)
                    policy.update_budget_al(params['size_init_train_data']*len(predicates)*crowd_votes_per_item_al)
                    SAL.screening_out_threshold = screening_out_threshold_machines
                    while policy.is_continue_al:
                        # SAL.update_stat()  # uncomment if use predicate selection feature
                        pr = SAL.select_predicate()
                        try:
                            query_idx = SAL.query(pr)
                        except:
                            # exit the loop if we crowdsourced all the items
                            break
                        # crowdsource sampled items
                        gt_items_queried = SAL.learners[pr].y_pool[query_idx]
                        y_crowdsourced = CrowdSimulator.crowdsource_items(item_ids_helper[pr][query_idx], gt_items_queried, pr,
                                                                          crowd_acc[pr], crowd_votes_per_item_al, crowd_votes_counts)
                        SAL.teach(pr, query_idx, y_crowdsourced)
                        item_ids_helper[pr] = np.delete(item_ids_helper[pr], query_idx)

                        policy.update_budget_al(SAL.n_instances_query*crowd_votes_per_item_al)

                    unclassified_item_ids = np.arange(items_num)
                    # Get prior from machines
                    for item_id in range(items_num):
                        prior_prob[item_id] = {}
                        for pr in predicates:
                            prediction = SAL.learners[pr].learner.predict_proba(vectorizer.transform([X[item_id]]))[0]
                            prior_prob[item_id][pr] = {'in': prediction[1], 'out': prediction[0]}
                    print('experiment_id {}'.format(experiment_id), end=', ')

                # if Available Budget for Crowd-Box DO SM-RUN
                if policy.B_crowd:
                    policy.B_crowd = policy.B - policy.B_al_spent
                    estimated_predicate_accuracy = {}
                    estimated_predicate_selectivity = {}
                    for pr in predicates:
                        estimated_predicate_accuracy[pr] = sum(crowd_acc[pr]) / 2
                        estimated_predicate_selectivity[pr] = sum(y_predicate[pr]) / len(y_predicate[pr])
                    smr_params = {
                        'estimated_predicate_accuracy': estimated_predicate_accuracy,
                        'estimated_predicate_selectivity': estimated_predicate_selectivity,
                        'predicates': predicates,
                        'item_predicate_gt': item_predicate_gt,
                        'clf_threshold': params['screening_out_threshold'],
                        'stop_score': params['stop_score'],
                        'crowd_acc': crowd_acc,
                        'prior_prob': prior_prob
                    }
                    SMR = ShortestMultiRun(smr_params)
                    unclassified_item_ids = np.arange(items_num)
                    # crowdsource items for SM-Run base-round in case poor SM-Run used
                    if switch_point == 0:
                        baseround_item_num = 50  # since 50 used in WWW2018 Krivosheev et.al
                        items_baseround = unclassified_item_ids[:baseround_item_num]
                        for pr in predicates:
                            gt_items_baseround = {item_id: item_predicate_gt[pr][item_id] for item_id in items_baseround}
                            CrowdSimulator.crowdsource_items(items_baseround, gt_items_baseround, pr, crowd_acc[pr],
                                                             crowd_votes_per_item_al, crowd_votes_counts)
                            policy.update_budget_crowd(baseround_item_num * crowd_votes_per_item_al)
                    unclassified_item_ids = SMR.classify_items(unclassified_item_ids, crowd_votes_counts, item_labels)

                    while policy.is_continue_crowd and unclassified_item_ids.any():
                        # Check money
                        if (policy.B_crowd - policy.B_crowd_spent) < len(unclassified_item_ids):
                            unclassified_item_ids = unclassified_item_ids[:(policy.B_crowd - policy.B_crowd_spent)]
                        unclassified_item_ids, budget_round = SMR.do_round(crowd_votes_counts, unclassified_item_ids, item_labels)
                        policy.update_budget_crowd(budget_round)
                    # print('Crowd-Box finished')

                # if budget is over and we did the AL part then classify the rest of the items via machines
                if unclassified_item_ids.any() and switch_point != 0:
                    predicted = SAL.predict(vectorizer.transform(X[unclassified_item_ids]))
                    item_labels.update(dict(zip(unclassified_item_ids, predicted)))

                # compute metrics and pint results to csv
                metrics = MetricsMixin.compute_screening_metrics(y_screening_dict, item_labels, params['lr'], params['beta'])
                pre, rec, f_beta, loss, fn_count, fp_count = metrics
                budget_spent_item = (policy.B_al_spent + policy.B_crowd_spent) / items_num
                results_list.append([budget_per_item, budget_spent_item, pre, rec, f_beta, loss, fn_count,
                                     fp_count, switch_point])

                print('budget spent per item: {:1.3f}, loss: {:1.3f}, fbeta: {:1.3f}, '
                      'recall: {:1.3f}, precisoin: {:1.3f}'
                      .format(budget_spent_item, loss, f_beta, rec, pre))
                print('--------------------------------------------------------------')

            df = pd.DataFrame(results_list, columns=['budget_per_item', 'budget_spent_per_item',
                                                     'precision', 'recall', 'f{}'.format(params['beta']), 'loss',
                                                     'fn_count', 'fp_count', 'AL_switch_point'])
            df = compute_mean_std(df)
            df['active_learning_strategy'] = params['sampling_strategy'].__name__ if switch_point != 0 else ''
            df['screening_out_threshold'] = params['screening_out_threshold']
            df_to_print = df_to_print.append(df, ignore_index=True)

    file_name = params['dataset_file_name'][:-4] + '_experiment_nums_{}_ninstq_{}'.format(params['experiment_nums'], params['n_instances_query'])
    if 'binary' in file_name:
        file_name = 'binary_' + file_name
    if os.path.isfile('../output/{}.csv'.format(file_name)):
        df_prev = pd.read_csv('../output/{}.csv'.format(file_name))
        df_new = df_prev.append(df_to_print, ignore_index=True)
        df_new.to_csv('../output/adaptive_machines_and_crowd/{}.csv'.format(file_name), index=False)
    else:
        df_to_print.to_csv('../output/{}.csv'.format(file_name), index=False)


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
                crowd_votes_counts[item_id][pr]['in'] = params['crowd_votes_per_item_al']
            else:
                crowd_votes_counts[item_id][pr]['out'] = params['crowd_votes_per_item_al']
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


def compute_mean_std(df):
    columns_mean = [c + '_mean' for c in df.columns]
    columns_median = [c + '_median' for c in df.columns]
    columns_std = [c + '_std' for c in df.columns]
    old_columns = df.columns.values
    df_mean = df.mean().rename(dict(zip(old_columns, columns_mean)))
    df_std = df.std().rename(dict(zip(old_columns, columns_std)))
    df_median = df.median().rename(dict(zip(old_columns, columns_median)))

    return pd.concat([df_mean, df_std, df_median])
