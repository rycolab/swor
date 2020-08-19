import logging
import time

import utils, sampling_utils
import numpy as np
from decoding.core import Decoder, PartialHypothesis


class ReferenceDecoder(Decoder):
    
    name = "reference"
    def __init__(self, decoder_args):
        """Creates a new reference decoder instance. The following values are
        fetched from `decoder_args`:
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(ReferenceDecoder, self).__init__(decoder_args)
        
    def decode(self, src_sentence, trgt_sentence):
        self.trgt_sentence = trgt_sentence + [utils.EOS_ID]
        self.initialize_predictor(src_sentence)

        hypo = PartialHypothesis(self.get_predictor_states())
        while hypo.get_last_word() != utils.EOS_ID:
            self._expand_hypo(hypo)
                
        hypo.score = self.get_adjusted_score(hypo)
        self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()


    def _expand_hypo(self,hypo):

        self.set_predictor_states(hypo.predictor_states)
        next_word = self.trgt_sentence[len(hypo.trgt_sentence)]
        ids, posterior, _ = self.apply_predictor()
        ind = utils.binary_search(ids, k)

        max_score = utils.max_(posterior)
        hypo.predictor_states = self.get_predictor_states()

        hypo.score += posterior[ind] 
        hypo.score_breakdown.append(posterior[ind])
        hypo.trgt_sentence += [next_word]
        self.consume(next_word)

class InclusionProbabilityDecoder(Decoder):
    
    name = "inclusion"
    def __init__(self, decoder_args):
        """Creates a new reference decoder instance. The following values are
        fetched from `decoder_args`:
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(InclusionProbabilityDecoder, self).__init__(decoder_args)
        self.nbest = decoder_args.nbest
        self.early_stopping = decoder_args.early_stopping
        
    def decode(self, src_sentence, trgt_sentence, seed=0):
        self.trgt_sentence = trgt_sentence + [utils.EOS_ID] if trgt_sentence[-1] != utils.EOS_ID else trgt_sentence
        self.initialize_predictor(src_sentence)

        it = 0
        self.beam_prob = 0.
        hypos = [PartialHypothesis(self.get_predictor_states())]
        trgt_hypo = hypos[0]

        while not self._all_eos(hypos) and it < self.max_len:
            it += 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    if hypo != trgt_hypo:
                        next_hypos.append(hypo)
                        next_scores.append(self.get_adjusted_score(hypo))
                    continue 
                for next_hypo in self._expand_hypo(hypo):
                    if hypo == trgt_hypo and next_hypo.trgt_sentence[-1] == self.trgt_sentence[len(trgt_hypo.trgt_sentence)]:
                        trgt_hypo = next_hypo
                        continue
                    next_scores.append(self.get_adjusted_score(next_hypo))
                    next_hypos.append(next_hypo)

            next_scores.append(self.get_adjusted_score(trgt_hypo))
            next_hypos.append(trgt_hypo)
            hypos = self._get_next_hypos(next_hypos, next_scores, seed)

        assert trgt_hypo.trgt_sentence == self.trgt_sentence
        assert self.beam_prob <= 0
        return trgt_hypo.base_score

    def _get_next_hypos(self, hypos, scores, seed):
        # faster to append to python list then convert to np array

        scores = np.array(scores)
        inds, cur_beam_prob, inc_probs = sampling_utils.log_sample_k_dpp(scores, 
                                                        self.nbest, seed=seed, 
                                                        include_last=True)
        assert len(inds) == min(len(scores), self.nbest)
        
        for i in inds:
            assert inc_probs[i] <= 0.
            hypos[i].base_score += inc_probs[i]
        self.beam_prob += cur_beam_prob
        return [hypos[ind] for ind in inds]

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        return all([hypo.get_last_word() == utils.EOS_ID for hypo in hypos])
                
