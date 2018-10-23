import numpy as np
from scipy.special import binom


class ShortestMultiRun:

    def __init__(self, params):
        # self.workers_accuracy = params['workers_accuracy']
        self.estimated_predicate_accuracy = params['estimated_predicate_accuracy']
        self.estimated_predicate_selectivity = params['estimated_predicate_selectivity']
        self.predicates = params['predicates']
        self.clf_threshold = params['clf_threshold']
        # self.ground_truth = params['ground_truth']

    def do_round(self, crowd_votes_counts, item_ids, item_labels):
        budget_round = None

        unclassified_item_ids = self.classify_items(item_ids, crowd_votes_counts, item_labels)

        return unclassified_item_ids, budget_round

    def classify_items(self, item_ids, crowd_votes_counts, item_labels):
        unclassified_item_ids = []

        for item_id in item_ids:
            prob_item_in = 1.
            for predicate in self.predicates:
                preducate_acc = self.estimated_predicate_accuracy[predicate]
                predicate_select = self.estimated_predicate_selectivity[predicate]

                if hasattr(self, 'prior_prob_pos'):
                    # TO DO: use machine prior!!!
                    prior_pred_in = predicate_select
                else:
                    prior_pred_in = predicate_select

                in_c, out_c = [crowd_votes_counts[item_id][predicate][key] for key in ['in', 'out']]
                if in_c == 0 and out_c == 0:
                    prob_predicate_in = predicate_select
                else:
                    term_in = binom(in_c + out_c, in_c) * preducate_acc ** in_c \
                               * (1 - preducate_acc) ** out_c * prior_pred_in
                    term_out = binom(in_c + out_c, out_c) * preducate_acc ** out_c \
                               * (1 - preducate_acc) ** in_c * (1 - prior_pred_in)
                    prob_predicate_in = term_in / (term_in + term_out)
                prob_item_in *= prob_predicate_in
            prob_item_out = 1 - prob_item_in

            if prob_item_out > self.clf_threshold:
                item_labels[item_id] = 0
            elif prob_item_in > self.clf_threshold:
                item_labels[item_id] = 1
            else:
                unclassified_item_ids.append(item_id)

        # return items_classified, items_to_classify
        return np.array(unclassified_item_ids)
