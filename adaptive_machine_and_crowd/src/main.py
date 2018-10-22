from modAL.uncertainty import uncertainty_sampling
from adaptive_machine_and_crowd.src.utils import random_sampling, objective_aware_sampling, mix_sampling
from adaptive_machine_and_crowd.src.heuristic import Heuristic

from adaptive_machine_and_crowd.src.experiment_handler import run_experiment

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
    'shuffling_num': reputation number of the whole experiment,
    'dataset_file_name ': file name of dataset,
    'predicates': predicates will be used in experiment,
    'B': budget available for classification,
    'B_al_prop': proportion of B for training machines (AL-Box)
    
'''


if __name__ == '__main__':
    # Parameters for active learners
    n_instances_query = 200
    n_queries = 100
    size_init_train_data = 20
    sampling_strategy = uncertainty_sampling

    # Classification parameters
    screening_out_threshold = 0.7
    beta = 3
    lr = 5

    # Experiment parameters
    shuffling_num = 50
    B = 500000
    B_al_prop = 0.5

    # # OHUSMED DATASET
    # dataset_file_name = 'ohsumed_C14_C23_1grams.csv'
    # predicates = ['C14', 'C23']

    # AMAZON DATASET
    predicates = ['is_negative', 'is_book']
    dataset_file_name = '100000_reviews_lemmatized.csv'

    # # LONELINESS SLR DATASET
    # predicates = ['oa_predicate', 'study_predicate']
    # dataset_file_name = 'loneliness-dataset-2018.csv'

    # # parameters for crowdsourcing simulation
    crowd_acc = {predicates[0]: [0.7, 1.],
                 predicates[1]: [0.7, 1.]}
    crowd_votes_per_item = 5

    params = {
        'dataset_file_name': dataset_file_name,
        'n_instances_query': n_instances_query,
        'size_init_train_data': size_init_train_data,
        'screening_out_threshold': screening_out_threshold,
        'beta': beta,
        'lr': lr,
        'shuffling_num': shuffling_num,
        'predicates': predicates,
        'sampling_strategy': sampling_strategy,
        'crowd_acc': crowd_acc,
        'crowd_votes_per_item': crowd_votes_per_item,
        'heuristic': Heuristic,
        'B': B, 'B_al_prop': B_al_prop
    }

    run_experiment(params)
    print('Done!')
