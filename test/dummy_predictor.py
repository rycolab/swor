import logging
import os

import utils
from predictors.core import Predictor

import numpy as np
from scipy.stats import entropy
import copy
import hashlib


class DummyPredictor(Predictor):
    """Predictor for using fairseq models."""

    def __init__(self, seed, dist_temperature=1.0, vocab_size=10, ):

        super(DummyPredictor, self).__init__()
        self.vocab_size = vocab_size
        self.rg = np.random.default_rng(seed=seed)
        self.num_dists = 1000
        self.model_temperature = dist_temperature
        # Create fake distributions with random number generator
        self.prob_dists = [self.rg.uniform(size=self.vocab_size) for i in range(self.num_dists)]

    def get_unk_probability(self, posterior):
        """Fetch posterior[utils.UNK_ID]"""
        return utils.common_get(posterior, utils.UNK_ID, utils.NEG_INF)
                
    def predict_next(self, prefix=None):
        hash_rep = str(self.src) + str(self.consumed if prefix is None else prefix)
        hash_key = int(hashlib.sha256(hash_rep.encode('utf-8')).hexdigest(), 16) 
        dist_key = hash_key % self.num_dists
        unnorm_posterior = copy.copy(self.prob_dists[dist_key])
        unnorm_posterior[utils.EOS_ID] += (len(self.consumed) - len(self.src))*unnorm_posterior.max()/2
        return utils.log_softmax(unnorm_posterior, temperature=self.model_temperature)
    
    def initialize(self, src_sentence):
        """Initialize source tensors, reset consumed."""
        self.src = src_sentence
        self.consumed =  []
   
    def consume(self, word):
        """Append ``word`` to the current history."""
        self.consumed.append(word)
    
    def get_empty_str_prob(self):
        return self.get_initial_dist()[utils.EOS_ID].item()

    def get_initial_dist(self):
        return self.predict_next(prefix=[])

    def get_state(self):
        """The predictor state is the complete history."""
        return self.consumed, [[]]
    
    def set_state(self, state):
        """The predictor state is the complete history."""
        consumed, inc_states = state
        self.consumed = consumed

    def is_equal(self, state1, state2):
        """Returns true if the history is the same """
        return state1[0] == state2[0]

