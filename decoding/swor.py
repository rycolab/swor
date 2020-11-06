import numpy as np
import time
import copy
import logging
from collections import defaultdict
from datastructures.sum_heap import SumHeap 

import utils
import sampling_utils
from decoding.core import Decoder, PartialHypothesis


class BasicSworDecoder(Decoder):
    name = "basic_swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(BasicSworDecoder, self).__init__(decoder_args)
        self.nbest = decoder_args.nbest
        self.early_stopping = decoder_args.early_stopping
        assert not self.gumbel
        
    def decode(self, src_sentence):
        self.initialize_predictor(src_sentence)
        self.covered_lprob = utils.NEG_INF

        while len(self.full_hypos) < self.nbest and self.samples_left():
            if np.exp(self.covered_lprob) >= 1.0 - utils.MACHINE_EPS:
                logging.warn("Samples cover 100% of probability. Behavior beyond this point is undefined")
            self.reset_predictor(src_sentence)
            hypo = PartialHypothesis(self.get_predictor_states())
            hypo, score = self._expand_hypo(hypo)
            self.add_full_hypo(hypo.generate_full_hypothesis())
            self.covered_lprob = utils.log_add(self.covered_lprob, score)
            
        logging.info("%d sentences covering %f probability" %(len(self.full_hypos), np.exp(self.covered_lprob)))
        return self.full_hypos

    def initialize_predictor(self, src_sentence):
        self.dists = MapDist()
        super().initialize_predictor(src_sentence)

    def _expand_hypo(self, hypo):
        if hypo.get_last_word() == utils.EOS_ID or len(hypo) == self.max_len:
            return hypo, 0.0
        #assert False
        prefix = tuple(hypo.trgt_sentence)
        if not prefix in self.dists:
            if self.start:
                # prefix has no longer previously been seen. One deep copy to get started
                hypo.predictor_states = copy.deepcopy(hypo.predictor_states)
                self.set_predictor_states(hypo.predictor_states)
                self.start = False
            if hypo.word_to_consume is not None:
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None

            ids, posterior, _ = self.apply_predictor()
            # assert not np.any(np.isnan(lprobabilities))
            self.dists.add_dist(prefix, ids, utils.log_softmax(posterior, self.temperature) , self.get_predictor_states())
            
        ids, lprobabilities, adjusted_lprobabilities, states = self.dists.get(prefix)
        hypo.predictor_states = states

        ind = adjusted_lprobabilities.sample()
        next_word = ids[ind]

        hypo.score += adjusted_lprobabilities[ind]
        hypo.score_breakdown.append(lprobabilities[ind])
        hypo.trgt_sentence += [next_word]
        hypo.word_to_consume = next_word
        hypo, score = self._expand_hypo(hypo)
        score += lprobabilities[ind] 
        self.dists.adjust(prefix, next_word, score)
        hypo.base_score = sum(hypo.score_breakdown)
        return hypo, score
         
    def reset_predictor(self, src_sentence):
        self.start = True
        self.predictor.initialize(src_sentence)

    def samples_left(self):
        if len(self.full_hypos) == 0:
            return True
        if self.early_stopping and np.exp(self.covered_lprob) >= 1.0 - utils.MACHINE_EPS:
            return False
        start_hash = tuple()
        _, _, adjusted_lprobabilities, _ = self.dists.get(start_hash)
        n, d = adjusted_lprobabilities.n, adjusted_lprobabilities.d
        return np.any(~np.isnan(adjusted_lprobabilities.S[d:d+n]) > utils.NEG_INF )

    def is_deterministic(self):
        return False


class SworDecoder(BasicSworDecoder):
    name = "swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(SworDecoder, self).__init__(decoder_args)

    def _expand_hypo(self, hypo):
        if hypo.get_last_word() == utils.EOS_ID or len(hypo) == self.max_len:
            return hypo, self.dists.marg(tuple(hypo.trgt_sentence))
        prefix = tuple(hypo.trgt_sentence)
        if not prefix in self.dists:
            if self.start:
                # prefix has no longer previously been seen. One deep copy to get started
                hypo.predictor_states = copy.deepcopy(hypo.predictor_states)
                self.set_predictor_states(hypo.predictor_states)
                self.start = False
            if hypo.word_to_consume is not None:
                self.consume(hypo.word_to_consume)
                hypo.word_to_consume = None
            
            ids, posterior, _ = self.apply_predictor()
            marg = self.dists.marg(prefix)
            lprobabilities = utils.log_softmax(posterior, self.temperature) + marg
            self.dists.add_dist(prefix, ids, lprobabilities, self.get_predictor_states())

        ids, lprobabilities, adjusted_lprobabilities, states = self.dists.get(prefix)
        hypo.predictor_states = states

        ind = adjusted_lprobabilities.sample()
        next_word = ids[ind]

        hypo.score += adjusted_lprobabilities[ind]
        hypo.score_breakdown.append(lprobabilities[ind])
        hypo.trgt_sentence += [next_word]
        hypo.word_to_consume = next_word
        hypo, final = self._expand_hypo(hypo)
        self.dists.adjust(prefix, next_word, final)
        hypo.base_score = hypo.score_breakdown[-1]
        return hypo, final


class MemEfficientSworDecoder(BasicSworDecoder):
    name = "mem_eff_swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(MemEfficientSworDecoder, self).__init__(decoder_args)

    
    def _expand_hypo(self, hypo):
        if hypo.get_last_word() == utils.EOS_ID or len(hypo) == self.max_len:
            return hypo, 0.0
        if hypo.word_to_consume is not None:
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None
        prefix = tuple(hypo.trgt_sentence)
        ids, posterior, _ = self.apply_predictor()
        lprobabilities = utils.log_softmax(posterior, self.temperature)
        adjusted_lprobabilities = self.adjust_probabilities(lprobabilities, prefix, ids)

        ind = sampling_utils.log_multinomial_sample(adjusted_lprobabilities)
        next_word = ids[ind]

        hypo.score += adjusted_lprobabilities[ind]
        hypo.score_breakdown.append(lprobabilities[ind])
        hypo.trgt_sentence += [next_word]
        hypo.word_to_consume = next_word
        hypo, score = self._expand_hypo(hypo)
        score += lprobabilities[ind] 
        self.ids[prefix][next_word] = utils.log_add(score, self.ids[prefix][next_word])
        hypo.base_score = sum(hypo.score_breakdown)
        return hypo, score
        

    def adjust_probabilities(self, lprobabilities, hash_rep, ids):
        lprobabilities = np.copy(lprobabilities)
        for k, val in self.ids[hash_rep].items():
            ind = utils.binary_search(ids, k)
            lprobabilities[ind] = utils.log_minus(lprobabilities[ind], val)
        return lprobabilities

    def initialize_predictor(self, src_sentence):
        self.ids = defaultdict(lambda: defaultdict(lambda: utils.NEG_INF))
        self.src_sentence = src_sentence
        super().initialize_predictor(self.src_sentence)

    def samples_left(self):
        if len(self.full_hypos) == 0:
            return True
        if self.early_stopping and np.exp(self.covered_lprob) >= 1.0 - utils.MACHINE_EPS:
            return False
        self.reset_predictor(self.src_sentence)
        ids, posterior, _ = self.apply_predictor()
        start_hash = tuple()
        lprobabilities = utils.log_softmax(posterior, self.temperature)
        adjusted_lprobabilities = self.adjust_probabilities(lprobabilities, start_hash, ids)
        return np.any(~np.isnan(adjusted_lprobabilities) > utils.NEG_INF )


class CPSworDecoder(Decoder):
    name = "cp_swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(CPSworDecoder, self).__init__(decoder_args)
        self.nbest = decoder_args.nbest
        self.early_stopping = decoder_args.early_stopping
        self.estimate_rounds = decoder_args.inc_prob_estimate_rounds
        self.sample_beam = decoder_args.sub_beam if decoder_args.sub_beam else self.nbest
        assert not self.gumbel
    
    def decode(self, src_sentence):
        self.initialize_predictor(src_sentence)
        self.covered_lprob = utils.NEG_INF
        
        it = 0
        self.beam_prob = 0.
        hypos = [PartialHypothesis(self.get_predictor_states())]

        while not self._all_eos(hypos) and it < self.max_len:
            it += 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(hypo.base_score)
                    continue 
                for next_hypo in self._expand_hypo(hypo, self.sample_beam):
                    next_scores.append(next_hypo.base_score)
                    next_hypos.append(next_hypo)
            hypos = self._get_next_hypos(next_hypos, next_scores)

        assert self.beam_prob <= 1
        return self.get_full_hypos_sorted(hypos)

    def _get_next_hypos(self, hypos, scores, include_last=False):
        # faster to append to python list then convert to np array
        scores = np.array(scores)
        inds, cur_beam_prob, inc_probs = CPSworDecoder.log_sample_k_dpp(scores, 
                                                        self.nbest,
                                                        include_last=include_last)
        assert len(inds) == min(len(scores), self.nbest)
        self.beam_prob += cur_beam_prob
        for i in inds:
            hypos[i].score += inc_probs[i]
        return [hypos[ind] for ind in inds]

    def _expand_hypo(self, hypo, limit=0, return_dist=False):
        """Get the best beam size expansions of ``hypo``.
        
        Args:
            hypo (PartialHypothesis): Hypothesis to expand
        
        Returns:
            list. List of child hypotheses
        """
        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None

        ids, posterior, original_posterior = self.apply_predictor(hypo, limit)
        #assert hypo.predictor_states != self.get_predictor_states()
        new_states = self.get_predictor_states()
        new_hypos = [hypo.cheap_expand(
                        trgt_word,
                        hypo.score,
                        base_score=posterior[idx] + hypo.base_score,
                        breakdown=posterior[idx],
                        states=new_states
                        ) for idx, trgt_word in enumerate(ids)]
        return new_hypos

    def get_inclusion_prob_estimate(self, src_sentence, trgt, **kwargs):
        if self.estimate_rounds == 1:
            return trgt.total_score
        estimates = []
        for i in range(self.estimate_rounds):
            self.initialize_predictor(src_sentence)
            np.random.seed(seed=self.seed*self.nbest+i)
            estimates.append(self.monte_carlo_inclusion_prob_estimate(src_sentence, trgt))
        # reset seed
        np.random.seed(seed=self.seed)
        return utils.logsumexp(estimates) - np.log(self.estimate_rounds)

    def monte_carlo_inclusion_prob_estimate(self, src_sentence, trgt):
        self.trgt_sentence = trgt.trgt_sentence + [utils.EOS_ID] if trgt.trgt_sentence[-1] != utils.EOS_ID else trgt.trgt_sentence

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
                        next_scores.append(hypo.base_score)
                    continue 
                for next_hypo in self._expand_hypo(hypo, self.sample_beam):
                    if hypo == trgt_hypo and next_hypo.trgt_sentence[-1] == self.trgt_sentence[len(trgt_hypo.trgt_sentence)]:
                        trgt_hypo = next_hypo
                        continue
                    next_scores.append(next_hypo.base_score)
                    next_hypos.append(next_hypo)

            next_scores.append(trgt_hypo.base_score)
            next_hypos.append(trgt_hypo)
            hypos = self._get_next_hypos(next_hypos, next_scores, include_last=True)

        assert trgt_hypo.trgt_sentence == self.trgt_sentence
        assert self.beam_prob <= 0
        return trgt_hypo.score

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        return all([hypo.get_last_word() == utils.EOS_ID for hypo in hypos])

    def is_deterministic(self):
        return False

    @staticmethod
    def log_sample_k_dpp(log_lambdas, k, include_last=False):
        N = len(log_lambdas)
        if k >= N:
            return range(N), 0., [0.]*N
        
        log_E = sampling_utils.log_elem_polynomials(log_lambdas, k)
        inc_probs = CPSworDecoder.inclusion_probs(log_lambdas, k, log_E)
        stab_test = log_lambdas[inc_probs > 0.]
        if not stab_test.size == 0:
            logging.warn("Experiencing some numerical instability.")
        J = []
        if include_last:
            J.append(N-1)
            N -= 1
            k -= 1
            if k == 0:
                return J, CPSworDecoder.log_beam_prob(log_lambdas, log_E, J), inc_probs

        for n in range(N,0,-1):
            u = np.random.uniform()
            thresh = log_lambdas[n-1] + log_E[k-1,n-1] - log_E[k,n]  
            if np.log(u) < thresh:
                J.append(n-1)
                k -= 1
                if k == 0:
                    break
        return J, CPSworDecoder.log_beam_prob(log_lambdas, log_E, J), inc_probs

    @staticmethod
    def log_beam_prob(log_lambdas, log_E, beam):
        if len(beam) != log_E.shape[0] - 1:
            return utils.NEG_INF
        return sum([log_lambdas[i] for i in beam]) - log_E[-1,-1]

    @staticmethod
    def inclusion_probs(log_lambdas, k, E=None):
        if E is None:
            E = sampling_utils.log_elem_polynomials(log_lambdas, k)

        k_, N = E.shape[0] - 1, E.shape[1] - 1
        assert k_ == k
        dv = np.full(N, utils.NEG_INF)
        d_E = np.full((k+1,N+1), utils.NEG_INF)
        d_E[k, N] = 0.
        for r in reversed(range(1,k+1)):
            for n in reversed(range(1,N+1)):
                d_E[r,n-1]   = utils.log_add(d_E[r,n-1], d_E[r,n])
                dv[n-1]     = utils.log_add(dv[n-1], d_E[r,n] + E[r-1,n-1])
                d_E[r-1,n-1] = utils.log_add(d_E[r-1,n-1], d_E[r,n] + log_lambdas[n-1])

        Z = E[k, len(log_lambdas)]
        return dv + log_lambdas - Z

    @staticmethod
    def add_args(parser):
        parser.add_argument("--inc_prob_estimate_rounds", default=1, type=int,
                        help="Number of rounds to use when creating inclusion probability"
                        "estimate")

class PSworDecoder(CPSworDecoder):
    name = "p_swor"
    def __init__(self, decoder_args):
        """Creates a new SWOR decoder instance. The following values are
        fetched from `decoder_args`:
        
            nbest (int): number of desired samples. 
            temperature (float): temperature for shifting probability distributions
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
        """
        super(PSworDecoder, self).__init__(decoder_args)

    def decode(self, src_sentence):
        self.initialize_predictor(src_sentence)
        desired_k = self.nbest#np.power(self.nbest, 1./self.max_len)
        
        it = 0
        hypos = [PartialHypothesis(self.get_predictor_states())]
        hypos[0].base_score = 1.
        while not self._all_eos(hypos) and it < self.max_len:
            it += 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    self.add_full_hypo(hypo.generate_full_hypothesis()) 
                    continue 
                expansions, dist = self._expand_hypo(hypo, return_dist=True)
                c = sampling_utils.get_const(dist + hypo.score, desired_k)
                c /= hypo.base_score
                c = np.power(c, 1./(self.max_len - len(hypo)))
                hypo.base_score *= c
                for next_hypo in expansions:
                    next_scores.append(next_hypo.score_breakdown[-1] + np.log(c))
                    next_hypos.append(next_hypo)

            hypos = self._get_next_hypos(next_hypos, next_scores)
            
        return self.get_full_hypos_sorted(hypos)
    

    def _get_next_hypos(self, hypos, scores):
        # faster to append to python list then convert to np array
        scores = np.array(scores)
        inds, inc_probs = PSworDecoder.log_sample_poisson(scores, 
                                                normalize=False)
        for i in inds:
            hypos[i].base_score += min(0.,inc_probs[i])
        return [hypos[ind] for ind in inds]

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        return all([hypo.get_last_word() == utils.EOS_ID for hypo in hypos])

    @staticmethod
    def log_sample_poisson(log_lambdas, k=1, normalize=True):
        J = []
        
        inc_probs = np.log(k) + log_lambdas 
        if normalize:
            inc_probs -= utils.logsumexp(log_lambdas)
        
        for i,l in enumerate(inc_probs):
            u = np.random.uniform() 
            if np.log(u) < l:
                J.append(i)
        return J, inc_probs

class MapDist(object):

    def __init__(self):
        self.dist_map = {}

    def __contains__(self, key):
        return tuple(key) in self.dist_map

    def add_dist(self, prefix, ids, dist, states):
        self.dist_map[prefix] = Dist(ids, dist, states)

    def adjust(self, prefix, next_word, val):
        self.dist_map[prefix].adjust(next_word, val)

    def get(self, prefix):
        return self.dist_map[prefix].values()

    def marg(self, prefix):
        if not prefix[:-1] in self.dist_map:
            return 0
        return self.dist_map[prefix[:-1]].get_current(prefix[-1])


class Dist(object):

    def __init__(self, ids, lprobabilities, predictor_states):
        self.ids = ids
        self.lprobabilities = SumHeap(lprobabilities, log_space=True)
        self.adjustments = SumHeap(np.full_like(lprobabilities, utils.NEG_INF, dtype=np.float64), log_space=True)
        self.predictor_states = copy.deepcopy(predictor_states)
        self.adjusted_lprobabilities = SumHeap(lprobabilities, log_space=True)
    
    def get_current(self, k):
        ind = utils.binary_search(self.ids, k)
        return self.adjusted_lprobabilities[ind]

    def adjust(self, k, val):
        ind = utils.binary_search(self.ids, k)
        self.adjustments[ind] = utils.log_add(self.adjustments[ind], val)
        self.adjusted_lprobabilities[ind] = utils.log_minus(self.lprobabilities[ind], self.adjustments[ind])

    def values(self):
        return self.ids, self.lprobabilities, self.adjusted_lprobabilities, self.predictor_states


