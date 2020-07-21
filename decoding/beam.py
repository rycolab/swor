"""Implementation of the beam search strategy """

import copy
import logging
import time

import utils
from decoding.core import Decoder, PartialHypothesis
import numpy as np


class BeamDecoder(Decoder):
    """This decoder implements standard beam search and several
    variants of it such as diversity promoting beam search and beam
    search with heuristic future cost estimates. This implementation
    supports risk-free pruning.
    """
    
    def __init__(self, decoder_args):
        """Creates a new beam decoder instance. The following values
        are fetched from `decoder_args`:
        
            hypo_recombination (bool): Activates hypo recombination 
            beam (int): Absolute beam size. A beam of 12 means
                        that we keep track of 12 active hypotheses
 
            diversity_factor (float): If this is set to a positive 
                                      value we add diversity promoting
                                      penalization terms to the partial
                                      hypothesis scores following Li
                                      and Jurafsky, 2016
            early_stopping (bool): If true, we stop when the best
                                   scoring hypothesis ends with </S>.
                                   If false, we stop when all hypotheses
                                   end with </S>. Enable if you are
                                   only interested in the single best
                                   decoding result. If you want to 
                                   create full 12-best lists, disable

        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(BeamDecoder, self).__init__(decoder_args)
        self.nbest = max(1, decoder_args.nbest)
        self.beam_size = decoder_args.beam if not self.gumbel else self.nbest
        self.sub_beam_size = decoder_args.sub_beam if decoder_args.sub_beam > 0 else self.beam_size 
        if decoder_args.early_stopping:
            self.stop_criterion = self._best_eos 
        else:
            self.stop_criterion = self._all_eos

        self.string_kernel_n = decoder_args.string_kernel_n
        self.string_kernel_decay = decoder_args.string_kernel_decay
        self.string_kernel_weight = decoder_args.string_kernel_weight

        assert not (decoder_args.diverse_prob_method and decoder_args.diverse_log_method)
        self.diverse_decoding = (self.string_kernel_weight > 0.)

        self.diverse_method = "original"
        if decoder_args.diverse_prob_method:
            self.diverse_method = "prob"
        elif decoder_args.diverse_log_method:
            self.diverse_method = "log"
        if self.diverse_decoding and decoder_args.early_stopping:
            logging.warn("Early stopping enabled with diverse decoding. full "
                        "diverse set may not be returned")
    
    def _best_eos(self, hypos):
        """Returns true if the best hypothesis ends with </S>"""
        ln_scores = [self.get_adjusted_score(hypo) for hypo in hypos]
        best_inds = utils.argmax_n(ln_scores, self.nbest)
        return all([hypos[ind].get_last_word() == utils.EOS_ID for ind in best_inds])
            
    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        return all([hypo.get_last_word() == utils.EOS_ID for hypo in hypos])


    def _get_next_hypos(self, all_hypos, all_scores):
        """Get hypos for the next iteration. """
        if self.diverse_decoding:
            inds, self.string_kernel_state = \
                utils.select_with_string_kernel_diversity(all_hypos, self.beam_size, self.string_kernel_n,
                                                          self.string_kernel_decay, self.string_kernel_weight,
                                                          self.string_kernel_state, method=self.diverse_method)
        else:
            inds = utils.argmax_n(all_scores, self.beam_size)
        return [all_hypos[ind] for ind in inds]
    
    def _get_initial_hypos(self):
        """Get the list of initial ``PartialHypothesis``. """
        return [PartialHypothesis(self.get_predictor_states())]
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.count = 0
        self.time = 0
        self.initialize_predictors(src_sentence)
        if self.diverse_decoding:
            self.string_kernel_state = {}
        hypos = self._get_initial_hypos()
        it = 0
        while not self.stop_criterion(hypos):
            if it > self.max_len: # prevent infinite loops
                break
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self.get_adjusted_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo, self.sub_beam_size):
                    next_hypos.append(next_hypo)
                    next_scores.append(self.get_adjusted_score(next_hypo))
            hypos = self._get_next_hypos(next_hypos, next_scores)
        for hypo in hypos:
            if hypo.get_last_word() == utils.EOS_ID:
                hypo.score = self.get_adjusted_score(hypo)
                self.add_full_hypo(hypo.generate_full_hypothesis()) 
        if not self.full_hypos:
            logging.warn("No complete hypotheses found")

        if len(self.full_hypos) < self.nbest:
            logging.warn("Adding incomplete hypotheses as candidates")
            for hypo in hypos:
                if hypo.get_last_word() != utils.EOS_ID:
                    hypo.score = self.get_adjusted_score(hypo)
                    self.add_full_hypo(hypo.generate_full_hypothesis()) 
            
        return self.get_full_hypos_sorted()
