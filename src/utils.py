import numpy as np
import pandas as pd
import warnings, random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

# related to CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted


# load and vectorize data
def load_vectorize_data(file_name, predicates, seed):
    df = pd.read_csv('../data/ohsumed_data/{}'.format(file_name))
    df_screening_pos = df.loc[df['Y'] == 1]
    df_screening_neg = df.loc[df['Y'] == 0]

    X_screening_pos = df_screening_pos['tokens'].values
    X_screening_neg = df_screening_neg['tokens'].values
    X = np.append(X_screening_pos, X_screening_neg)

    y_screening = np.append(np.ones(X_screening_pos.shape[0]),
                            np.zeros(X_screening_neg.shape[0]))

    y_predicate = {}  # gt labels per predicate
    for pr in predicates:
        y_predicate[pr] = np.append(df_screening_pos[pr].values,
                                    df_screening_neg[pr].values)

    # vectorize and transform text
    vectorizer = TfidfVectorizer(lowercase=False, max_features=2000, ngram_range=(1, 1))
    X = vectorizer.fit_transform(X).toarray()

    # shuffle X, y in unison
    np.random.seed(seed)
    idx = np.random.permutation(y_screening.shape[0])
    X, y_screening = X[idx], y_screening[idx]
    for pr in predicates:
        y_predicate[pr] = y_predicate[pr][idx]

    return X, y_screening, y_predicate


# random sampling strategy for modAL
def random_sampling(_, X, n_instances=1, seed=123):
    random.seed(seed)
    query_idx = np.array(random.sample(range(X.shape[0]), n_instances))

    return query_idx, X[query_idx]


# positive class certainty sampling strategy for modAL
def positive_certainty_sampling(classifier, X, n_instances=1, **predict_proba_kwargs):
    prob_in = classifier.predict_proba(X, **predict_proba_kwargs)[:, 1]
    query_idx = np.argpartition(prob_in, -n_instances)[-n_instances:]

    return query_idx, X[query_idx]


# sampling takes into account conjunctive expression of predicates
def objective_aware_sampling(classifier, X, learners_, n_instances=1, **uncertainty_measure_kwargs):
    from modAL.uncertainty import classifier_uncertainty, multi_argmax
    uncertainty = classifier_uncertainty(classifier, X, **uncertainty_measure_kwargs)
    l_prob_in = np.ones(X.shape[0])
    if learners_:
        for l in learners_.values():
            l_prob_in *= l.learner.predict_proba(X)[:, 1]
        uncertainty_weighted = np.power(l_prob_in * uncertainty, 1 / (len(learners_) + 1))
    else:
        uncertainty_weighted = uncertainty

    query_idx = multi_argmax(uncertainty_weighted, n_instances=n_instances)

    return query_idx, X[query_idx]


# add classification threshold to CalibratedClassifierCV
class CalibratedClassifierCV(CalibratedClassifierCV):

    def __init__(self, base_estimator=None, method='sigmoid', cv=3, p_out=0.5):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.p_out = p_out

    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """

        check_is_fitted(self, ["classes_", "calibrated_classifiers_"])
        return np.array([0 if p > self.p_out else 1 for p in self.predict_proba(X)[:, 0]])


# screening metrics, aimed to obtain high recall
class MetricsMixin:

    @staticmethod
    def compute_screening_metrics(gt, predicted, lr):
        '''
        FP == False Inclusion
        FN == False Exclusion
        '''
        fp = 0.
        fn = 0.
        tp = 0.
        tn = 0.
        for gt_val, pred_val in zip(gt, predicted):
            if gt_val and not pred_val:
                fn += 1
            if not gt_val and pred_val:
                fp += 1
            if gt_val and pred_val:
                tp += 1
            if not gt_val and not pred_val:
                tn += 1
        loss = (fn * lr + fp) / len(gt)
        try:
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            beta = 1. / lr
            fbeta = (beta ** 2 + 1) * precision * recall / (recall + beta ** 2 * precision)
        except ZeroDivisionError:
            warnings.warn('ZeroDivisionError -> recall, precision, fbeta = 0., 0., 0')
            recall, precision, fbeta = 0., 0., 0

        return precision, recall, fbeta, loss

    @staticmethod
    def compute_tpr_tnr(gt, predicted):
        tn, fp, fn, tp = confusion_matrix(gt, predicted).ravel()
        TPR = tp / (tp + fn)  # sensitivity, recall, or true positive rate
        TNR = tn / (tn + fp)  # specificity or true negative rate

        return TPR, TNR


# transfrom data from k-fold CV and print results in csv
def transform_print(data_df, sampl_strategy, predicates):
    # compute mean and std, and median over k-fold cross validation results
    df_concat = pd.concat(data_df)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
    df_std = by_row_index.std()
    df_median = by_row_index.median()

    # form dataframe for printing out in csv
    df_to_print = df_means
    df_to_print.columns = ['num_items_queried', 'precision_mean',
                           'recall_mean', 'f_beta_mean', 'loss_mean']

    df_to_print['precision_median'] = df_median['precision']
    df_to_print['recall_median'] = df_median['recall']
    df_to_print['f_beta_median'] = df_median['f_beta']
    df_to_print['loss_median'] = df_median['loss']

    df_to_print['precision_std'] = df_std['precision']
    df_to_print['recall_std'] = df_std['recall']
    df_to_print['f_beta_std'] = df_std['f_beta']
    df_to_print['loss_std'] = df_std['loss']

    df_to_print['sampling_strategy'] = sampl_strategy
    df_to_print.to_csv('../data/multi_classifier_al/screening_al_{}_{}.csv'
                       .format(predicates[0], predicates[1]), index=False)
