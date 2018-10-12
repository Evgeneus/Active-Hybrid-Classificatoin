from machine_and_experts_annotate.src.experiment_handler import experiment_handler

'''
    Parameters for active learners:
    'n_instances_query': num of instances for labeling for 1 query,
    'n_queries': num of active learning iterations,
    'init_train_size': initial size of training dataset
    
    Classification parameters:
    'screening_out_threshold': threshold to classify a document OUT,
    'beta': beta for F_beta score,
    'lr': loss ration for the screening loss
    
    Experiment parameters:
    'test_size': proportion of test size,
    'k': reputation number of the whole experiment,
    'file_name': file name of dataset,
    'predicates': predicates will be used in experiment
    
'''


if __name__ == '__main__':
    # Parameters for active learners
    n_instances_query = 200
    n_queries = 100
    init_train_size = 20

    # Classification parameters
    screening_out_threshold = 0.7
    beta = 3
    lr = 5

    # Experiment parameters
    test_size = 0.4
    k = 5

    # OHUSMED DATASET
    file_name = 'ohsumed_C14_C23_1grams.csv'
    predicates = ['C14', 'C23']

    # AMAZON DATASET
    # predicates = ['is_negative', 'is_book']
    # file_name = '100000_reviews_lemmatized.csv'

    # LONELINESS SLR DATASET
    # predicates = ['oa_predicate', 'study_predicate']
    # file_name = 'loneliness-dataset-2018.csv'

    experiment_handler((file_name,
                       n_instances_query,
                       n_queries,
                       init_train_size,
                       screening_out_threshold,
                       test_size,
                       beta, lr, k,
                       predicates))

    print('Done!')
