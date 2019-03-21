import sys, os
# path_to_project = os.path.realpath('main.py')[:-38]
path_to_project = os.path.realpath('main.py')[:-38]
# path_to_project = '/home/evgeny.krivosheev/Active-Hybrid-Classificatoin_MultiPredicate/'
sys.path.append(path_to_project)

from modAL.uncertainty import uncertainty_sampling
from adaptive_machine_and_crowd.src.utils import random_sampling, objective_aware_sampling

from adaptive_machine_and_crowd.src.experiment_handler import run_experiment
import numpy as np

'''
    Parameters for active learners:
    'n_instances_query': num of instances for labeling for 1 query,
    'size_init_train_data': initial size of training dataset,
    'sampling_strategies': list of active learning sampling strategies
    
    Classification parameters:
    'screening_out_threshold': threshold to classify a document OUT,
    'beta': beta for F_beta score,
    'lr': loss ration for the screening loss
    
    Experiment parameters:
    'experiment_nums': reputation number of the whole experiment,
    'dataset_file_name ': file name of dataset,
    'predicates': predicates will be used in experiment,
    'B': budget available for classification,
    'B_al_prop': proportion of B for training machines (AL-Box)
'''


if __name__ == '__main__':
    # datasets = 'amazon', 'ohusmed', 'slr', 'amazon_binary', 'ohusmed_binary', 'slr_binary'
    dataset = 'amazon'
    if dataset == 'amazon':
        # AMAZON DATASET
        predicates = ['is_negative', 'is_book']
        dataset_file_name = '5000_reviews_lemmatized.csv'
        dataset_size = 5000
        crowd_acc = {predicates[0]: [0.94, 0.94], predicates[1]: [0.94, 0.94]}
    elif dataset == 'ohusmed':
        # OHUSMED DATASET
        dataset_file_name = 'ohsumed_C14_C23_1grams.csv'
        predicates = ['C14', 'C23']
        dataset_size = 34387
        crowd_acc = {predicates[0]: [0.6, 1.], predicates[1]: [0.6, 1.]}
    elif dataset == 'slr':
        # LONELINESS SLR DATASET
        predicates = ['oa_predicate', 'study_predicate']
        dataset_file_name = 'loneliness-dataset-2018.csv'
        dataset_size = 825
        crowd_acc = {predicates[0]: [0.8, 0.8], predicates[1]: [0.6, 0.6]}
    elif dataset == 'amazon_binary':
        # AMAZON BINARY DATASET
        predicates = ['Y']
        dataset_file_name = '5000_reviews_lemmatized.csv'
        dataset_size = 5000
        crowd_acc = {predicates[0]: [0.94, 0.94]}
    elif dataset == 'slr_binary':
        # LONELINESS BINARY SLR DATASET
        predicates = ['Y']
        dataset_file_name = 'loneliness-dataset-2018.csv'
        dataset_size = 825
        crowd_acc = {predicates[0]: [0.75, 0.75]}
    elif dataset == 'ohusmed_binary':
        # OHUSMED BINARY DATASET
        dataset_file_name = 'ohsumed_C14_C23_1grams.csv'
        predicates = ['Y']
        dataset_size = 34387
        crowd_acc = {predicates[0]: [0.6, 1.]}
    else:
        exit(1)

    # Parameters for active learners
    n_instances_query = 100
    size_init_train_data = 20

    # Classification parameters
    screening_out_threshold = 0.99  # for SM-Run and ML
    stop_score = 50  # for SM-Run Algorithm
    beta = 1
    lr = 5

    # Experiment parameters
    experiment_nums = 10
    policy_switch_point = [0.55]  # must spent no more than 0,55 on AL
    budget_per_item = np.arange(1, 9, 1)  # number of votes per item we can spend per item on average
    crowd_votes_per_item_al = 3  # for Active Learning annotation

    for sampling_strategy in [uncertainty_sampling]:
        print('{} is Running!'.format(sampling_strategy.__name__))
        params = {
            'dataset_file_name': dataset_file_name,
            'n_instances_query': n_instances_query,
            'size_init_train_data': size_init_train_data,
            'screening_out_threshold': screening_out_threshold,
            'beta': beta,
            'lr': lr,
            'experiment_nums': experiment_nums,
            'predicates': predicates,
            'sampling_strategy': sampling_strategy,
            'crowd_acc': crowd_acc,
            'crowd_votes_per_item_al': crowd_votes_per_item_al,
            'policy_switch_point': policy_switch_point,
            'budget_per_item': budget_per_item,
            'stop_score': stop_score,
            'dataset_size': dataset_size,
            'path_to_project' : path_to_project
        }

        run_experiment(params)
        print('{} is Done!'.format(sampling_strategy.__name__))
