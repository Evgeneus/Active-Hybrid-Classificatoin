import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
# from modAL.uncertainty import uncertainty_sampling

from src.utils import load_vectorize_data, transform_print, objective_aware_sampling
from src.active_learning import Learner, ScreeningActiveLearner

seed = 123


if __name__ == '__main__':
    predicates = ['C04', 'C12']
    file_name = 'ohsumed_C04_C12_1grams.csv'
    # load and transform data
    X, y_screening, y_predicate = load_vectorize_data(file_name, predicates, seed)

    data_df = []
    k = 10
    skf = StratifiedKFold(n_splits=k, random_state=seed)
    for train_idx, test_idx in skf.split(X, y_screening):
        print('-------------------------------')
        # split training-test datasets
        X_train, X_test = X[train_idx], X[test_idx]
        y_screening_test = y_screening[test_idx]
        y_predicate_train, y_predicate_test = {}, {}
        for pr in predicates:
            y_predicate_train[pr] = y_predicate[pr][train_idx]
            y_predicate_test[pr] = y_predicate[pr][test_idx]

        # dict of active learners per predicate
        learners = {}
        for pr in predicates:  # setup predicate-based learners
            params = {
                'clf': CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=seed)),
                'undersampling_thr': 0.333,
                'seed': seed,
                'init_train_size': 10,
                'sampling_strategy': objective_aware_sampling,
                'p_out': 0.5,
            }
            learner = Learner(params)
            learner.setup_active_learner(X_train, y_predicate_train[pr], X_test, y_predicate_test[pr])
            learners[pr] = learner

        screening_params = {
            'n_instances_query': 50,  # num of instances for labeling for 1 query
            'seed': seed,
            'init_train_size': 10,
            'p_out': 0.65,
            'lr': 10,
            'learners': learners
        }
        SAL = ScreeningActiveLearner(screening_params)
        SAL.init_stat()  # initialize statistic for predicates
        n_queries = 80
        num_items_queried = params['init_train_size']*len(predicates)
        data = []
        for i in range(n_queries):
            SAL.update_stat()

            pr = SAL.select_predicate(i)
            query_idx = SAL.query(pr)
            SAL.teach(pr, query_idx)
            predicted = SAL.predict(X_test)
            metrics = SAL.compute_screening_metrics(y_screening_test, predicted, SAL.lr)

            pre, rec, fbeta, loss = metrics
            num_items_queried += SAL.n_instances_query
            data.append([num_items_queried, pre, rec, fbeta, loss])

            print('query no. {}: loss: {:1.3f}, fbeta: {:1.3f}, '
                          'recall: {:1.3f}, precisoin: {:1.3f}'
                  .format(i + 1, loss, fbeta, rec, pre))

            data_df.append(pd.DataFrame(data, columns=['num_items_queried',
                                                       'precision', 'recall',
                                                       'f_beta', 'loss']))

    transform_print(data_df, params['sampling_strategy'].__name__, predicates)
    print('Done!')
