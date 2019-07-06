import numpy as np
from scipy.special import binom
import random
from heapq import nsmallest


class ShortestMultiRun:

    def __init__(self, params):
        self.estimated_predicate_accuracy = params['estimated_predicate_accuracy']
        self.estimated_predicate_selectivity = params['estimated_predicate_selectivity']
        self.predicates = params['predicates']
        self.clf_threshold = params['clf_threshold']
        self.stop_score = params['stop_score']
        self.crowd_acc_range = params['crowd_acc']
        self.item_predicate_gt = params['item_predicate_gt']
        self.prior_prob = params.get('prior_prob', None)
        self.max_votes_per_item = 20
        self.difficult_item_ids = params['difficult_item_ids']
        self.diff = 0.2
        self.crowd_votes_item_avg = params['crowd_votes_item_avg']
        self.items_round = params['items_round']

    def do_round(self, crowd_votes_counts, unclassified_item_ids, item_labels):
        budget_round = 0
        for _ in range(self.crowd_votes_item_avg):
            predicate_assigned, item_score = self.assign_predicates(unclassified_item_ids, crowd_votes_counts)
            item_ids_tocrowd = nsmallest(self.items_round, item_score, key=item_score.get)
            if len(item_ids_tocrowd) > 0:
                unclassified_item_ids = set(predicate_assigned.keys()) - set(item_ids_tocrowd)
                predicate_assigned_round = {id_: predicate_assigned[id_] for id_ in item_ids_tocrowd}
                self.crowdsource_items(crowd_votes_counts, predicate_assigned_round)
                unclassified_item_ids |= set(self.classify_items(predicate_assigned_round.keys(), crowd_votes_counts, item_labels))
                unclassified_item_ids = list(unclassified_item_ids)
            else:
                return [], budget_round
            budget_round += len(item_ids_tocrowd)

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

    def assign_predicates(self, item_ids, crowd_votes_counts):
        predicate_assigned = {}
        item_score = {}
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
            if classify_score[predicate_best_score] < self.stop_score and crowdsourced_votes_num < self.max_votes_per_item:
                predicate_assigned[item_id] = predicate_best_score
                item_score[item_id] = classify_score[predicate_best_score]

        return predicate_assigned, item_score

    def crowdsource_items(self, crowd_votes_counts, predicate_assigned):
        for item_id in predicate_assigned.keys():
            predicate = predicate_assigned[item_id]
            crowd_acc_range = self.crowd_acc_range[predicate]
            worker_acc = random.uniform(crowd_acc_range[0], crowd_acc_range[1])
            gt = self.item_predicate_gt[predicate][item_id]
            # reduce worker_acc if item is a difficult one
            if item_id in self.difficult_item_ids:
                worker_acc -= self.diff
            worker_vote = np.random.binomial(1, worker_acc if gt == 1 else 1 - worker_acc)
            if worker_vote == 1:
                crowd_votes_counts[item_id][predicate]['in'] += 1
            else:
                crowd_votes_counts[item_id][predicate]['out'] += 1

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
