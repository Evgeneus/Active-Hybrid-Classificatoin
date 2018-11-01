import numpy as np
from scipy.special import binom
import random


class ShortestMultiRun:

    def __init__(self, params):
        self.estimated_predicate_accuracy = params['estimated_predicate_accuracy']
        self.estimated_predicate_selectivity = params['estimated_predicate_selectivity']
        self.predicates = params['predicates']
        self.clf_threshold = params['clf_threshold']
        self.stop_score = params['stop_score']
        self.item_predicate_gt = params['item_predicate_gt']
        self.prior_prob = params.get('prior_prob', None)
        self.max_votes_per_item = 20

    def do_round(self, crowd_votes, crowd_votes_counts, item_ids, item_labels):
        predicate_assigned = self.assign_predicates(crowd_votes, item_ids, crowd_votes_counts)
        budget_round = self.crowdsource_items(crowd_votes, crowd_votes_counts, predicate_assigned)
        unclassified_item_ids = self.classify_items(predicate_assigned.keys(), crowd_votes_counts, item_labels)

        return unclassified_item_ids, budget_round

    def classify_items(self, item_ids, crowd_votes_counts, item_labels):
        unclassified_item_ids = []
        for item_id in item_ids:
            prob_item_in = 1.
            for predicate in self.predicates:
                prob_predicate_in = self._prob_predicate_in(predicate, item_id, crowd_votes_counts)
                prob_item_in *= prob_predicate_in
            prob_item_out = 1 - prob_item_in

            if prob_item_out > self.clf_threshold:
                item_labels[item_id] = 0
            elif prob_item_in > self.clf_threshold:
                item_labels[item_id] = 1
            else:
                unclassified_item_ids.append(item_id)

        return np.array(unclassified_item_ids)

    def assign_predicates(self, crowd_votes, item_ids, crowd_votes_counts):
        predicate_assigned = {}
        for item_id in item_ids:
            crowdsourced_votes_num = 0
            classify_score = {}
            joint_prob_votes_out = {predicate: 1. for predicate in self.predicates}
            for predicate in self.predicates:
                crowdsourced_votes_num += sum(crowd_votes_counts[item_id][predicate].values())
                # set up prob_item
                _prob_item_in = 1.
                for pr in [pr for pr in self.predicates if pr != predicate]:
                    _prob_item_in *= self._prob_predicate_in(pr, item_id, crowd_votes_counts)

                preducate_acc = self.estimated_predicate_accuracy[predicate]
                predicate_select = self.estimated_predicate_selectivity[predicate]
                if self.prior_prob:
                    prior_pred_in = self.prior_prob[item_id][predicate]['in']
                    prob_pred_out = 1 - prior_pred_in
                else:
                    prob_pred_out = 1 - predicate_select
                    prior_pred_in = predicate_select
                in_c, out_c = [crowd_votes_counts[item_id][predicate][key] for key in ['in', 'out']]
                for n in range(1, 11):
                    prob_next_vote_out = preducate_acc * prob_pred_out + (1 - preducate_acc) * (1 - prob_pred_out)
                    joint_prob_votes_out[predicate] *= prob_next_vote_out

                    term_in = binom(in_c + out_c + n, in_c) * preducate_acc ** in_c \
                              * (1 - preducate_acc) ** (out_c + n) * prior_pred_in
                    term_out = binom(in_c + out_c + n, out_c + n) * preducate_acc ** (out_c + n) \
                               * (1 - preducate_acc) ** in_c * (1 - prior_pred_in)

                    prob_predicate_in = term_in / (term_in + term_out)
                    prob_item_out = 1 - _prob_item_in * prob_predicate_in
                    if prob_item_out >= self.clf_threshold:
                        classify_score[predicate] = n / joint_prob_votes_out[predicate]
                        break
                    elif n == 10:
                        classify_score[predicate] = n / joint_prob_votes_out[predicate]

            predicate_best_score = min(classify_score, key=classify_score.get)
            if classify_score[predicate_best_score] < self.stop_score \
                    and crowdsourced_votes_num < self.max_votes_per_item\
                    and crowd_votes[item_id][predicate_best_score]:
                predicate_assigned[item_id] = predicate_best_score

        return predicate_assigned

    def crowdsource_items(self, crowd_votes, crowd_votes_counts, predicate_assigned):
        votes_num = 0
        for item_id in predicate_assigned.keys():
            predicate = predicate_assigned[item_id]
            vote_list = crowd_votes[item_id][predicate]
            if vote_list:
                worker_vote = vote_list.pop()
                votes_num += 1
                if worker_vote == 1:
                    crowd_votes_counts[item_id][predicate]['in'] += 1
                else:
                    crowd_votes_counts[item_id][predicate]['out'] += 1
        return votes_num

    def _prob_predicate_in(self, predicate, item_id, crowd_votes_counts):
        preducate_acc = self.estimated_predicate_accuracy[predicate]
        predicate_select = self.estimated_predicate_selectivity[predicate]

        if self.prior_prob:
            prior_pred_in = self.prior_prob[item_id][predicate]['in']
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

        return prob_predicate_in
