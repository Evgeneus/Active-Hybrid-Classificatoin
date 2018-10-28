from modAL.uncertainty import uncertainty_sampling
from adaptive_machine_and_crowd.src.utils import random_sampling, objective_aware_sampling, mix_sampling

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
    # Parameters for active learners
    n_instances_query = 100
    size_init_train_data = 20
    sampling_strategy = objective_aware_sampling

    # Classification parameters
    screening_out_threshold = 0.99  # for SM-Run
    stop_score = 50  # for SM-Run Algorithm
    beta = 3
    lr = 5

    # # OHUSMED DATASET
    # dataset_file_name = 'ohsumed_C14_C23_1grams.csv'
    # predicates = ['C14', 'C23']
    # dataset_size = 34387
    # crowd_acc = {predicates[0]: [0.6, 1.],
    #              predicates[1]: [0.6, 1.]}

    # # AMAZON DATASET
    predicates = ['is_negative', 'is_book']
    dataset_file_name = '5000_reviews_lemmatized.csv'
    dataset_size = 5000
    crowd_acc = {predicates[0]: [0.96, 0.96],
                 predicates[1]: [0.96, 0.96]}

    # # LONELINESS SLR DATASET
    # predicates = ['oa_predicate', 'study_predicate']
    # dataset_file_name = 'loneliness-dataset-2018.csv'
    # dataset_size = 586


    # Experiment parameters
    experiment_nums = 10
    policy_switch_point = np.arange(0., 0.8, 0.1)
    budget_per_item = np.arange(1, 9, 1)  # number of votes per item we can spend per item on average
    dget_per_item = [3, 6]  # number of votes per item we can spend per item on average
    crowd_votes_per_item_al = 3  # for Active Learning annotation

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
        'dataset_size': dataset_size
    }

    run_experiment(params)
    print('Done!')
