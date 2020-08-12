import statistics
import time
from abc import abstractmethod
import operator
import logging
import os
import sys
from bisect import bisect_left
from functools import reduce
from itertools import product
import multiprocessing
from string_kernel_utils import get_string_kernel_value_to_subtract_compare_with_lodhi_recursive_normalized

import numpy as np
from scipy.special import logsumexp

# Reserved IDs
GO_ID = 1
"""Reserved word ID for the start-of-sentence symbol. """


EOS_ID = 2
"""Reserved word ID for the end-of-sentence symbol. """


UNK_ID = 0
"""Reserved word ID for the unknown word (UNK). """


NEG_INF = -np.inf


MACHINE_EPS = np.finfo(float).eps


LOG_MACHINE_EPS = np.log(MACHINE_EPS)


INF = np.inf


EPS_P = 0.00001


# set TEST = True if string kernel should be tested
TEST = False


def switch_to_fairseq_indexing():
    """Calling this method overrides the global definitions of the 
    reserved  word ids ``GO_ID``, ``EOS_ID``, and ``UNK_ID``
    with the fairseq indexing scheme. 
    """
    global GO_ID
    global EOS_ID
    global UNK_ID
    GO_ID = 0
    EOS_ID = 2
    UNK_ID = 3


def switch_to_t2t_indexing():
    """Calling this method overrides the global definitions of the 
    reserved  word ids ``GO_ID``, ``EOS_ID``, and ``UNK_ID``
    with the tensor2tensor indexing scheme. This scheme is used in all
    t2t models. 
    """
    global GO_ID
    global EOS_ID
    global UNK_ID
    GO_ID = 2 # Usually not used
    EOS_ID = 1
    UNK_ID = 3 # Don't rely on this: UNK not standardized in T2T


# Log summation


def log_sum_tropical_semiring(vals):
    """Approximates summation in log space with the max.
    
    Args:
        vals  (set): List or set of numerical values
    """
    return max(vals)


def log_sum_log_semiring(vals):
    """Uses the ``logsumexp`` function in scipy to calculate the log of
    the sum of a set of log values.
    
    Args:
        vals  (set): List or set of numerical values
    """
    return logsumexp(np.asarray([val for val in vals]))


log_sum = log_sum_log_semiring
"""Defines which log summation function to use. """


def oov_to_unk(seq, vocab_size, unk_idx=None):
    if unk_idx is None:
        unk_idx = UNK_ID
    return [x if x < vocab_size else unk_idx for x in seq]

# Maximum functions

def argmax_n(arr, n):
    """Get indices of the ``n`` maximum entries in ``arr``. The 
    parameter ``arr`` can be a dictionary. The returned index set is 
    not guaranteed to be sorted.
    
    Args:
        arr (list,array,dict):  Set of numerical values
        n  (int):  Number of values to retrieve
    
    Returns:
        List of indices or keys of the ``n`` maximum entries in ``arr``
    """
    if isinstance(arr, dict):
        return sorted(arr, key=arr.get, reverse=True)[:n]
    elif len(arr) <= n:
        return range(len(arr))
    elif hasattr(arr, 'is_cuda') and arr.is_cuda:
        return np.argpartition(arr.cpu(), -n)[-n:]
    return np.argpartition(arr, -n)[-n:]


def max_(arr):
    """Get indices of the ``n`` maximum entries in ``arr``. The 
    parameter ``arr`` can be a dictionary. The returned index set is 
    not guaranteed to be sorted.
    
    Args:
        arr (list,array,dict):  Set of numerical values
        n  (int):  Number of values to retrieve
    
    Returns:
        List of indices or keys of the ``n`` maximum entries in ``arr``
    """
    if isinstance(arr, dict):
        return max(arr.values())
    if isinstance(arr, list):
        return max(arr)
    return np.max(arr)


def argmax(arr):
    """Get the index of the maximum entry in ``arr``. The parameter can
    be a dictionary.
    
    Args:
        arr (list,array,dict):  Set of numerical values
    
    Returns:
        Index or key of the maximum entry in ``arr``
    """
    if isinstance(arr, dict):
        return max(arr.items(), key=operator.itemgetter(1))[0]
    else:
        return np.argmax(arr)

def logmexp(x):
    return np.log1p(-np.exp(x))

def logpexp(x):
    return np.log1p(np.exp(x))


def logsigmoid(x):
    """
    log(sigmoid(x)) = -log(1+exp(-x)) = -log1pexp(-x)
    """
    return -log1pexp(-x)


def log1pexp(x):
    """
    Numerically stable implementation of log(1+exp(x)) aka softmax(0,x).

    -log1pexp(-x) is log(sigmoid(x))

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x <= -37:
        return np.exp(x)
    elif -37 <= x <= 18:
        return np.log1p(np.exp(x))
    elif 18 < x <= 33.3:
        return x + np.exp(-x)
    else:
        return x


def log1mexp(x):
    """
    Numerically stable implementation of log(1-exp(x))

    Note: function is finite for x < 0.

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x >= 0:
        return np.nan
    else:
        a = abs(x)
        if 0 < a <= 0.693:
            return np.log(-np.expm1(-a))
        else:
            return np.log1p(-np.exp(-a))


def log_add(x, y):
    # implementation: need separate checks for inf because inf-inf=nan.
    if x == NEG_INF:
        return y
    elif y == NEG_INF:
        return x
    else:
        if y <= x:
            d = y-x
            r = x
        else:
            d = x-y
            r = y
        return r + log1pexp(d)

vec_log_add = np.vectorize(log_add)

def log_minus(x, y):
    if x == y:
        return NEG_INF
    if y > x:
        if y-x > MACHINE_EPS:
            logging.warn("Using function log_minus for invalid values")
        return np.nan
    else:
        return x + log1mexp(y-x)


def log_add_old(a, b):
    # takes two log probabilities; equivalent to adding probabilities in log space
    if a == NEG_INF or b == NEG_INF:
        return max(a, b)
    smaller = min(a,b)
    larger = max(a,b)
    return larger + log1pexp(smaller - larger)

def log_minus_old(a, b):
    # takes two log probabilities; equivalent to subtracting probabilities in log space
    assert b <= a
    if a == b:
        return NEG_INF
    if b == NEG_INF:
        return a
    comp = a + log1mexp(-(a-b))
    return comp if not np.isnan(comp) else NEG_INF


def softmax(x, temperature=1.):
    return np.exp(log_softmax(x, temperature=temperature))

def log_softmax(x, temperature=1.):
    x = x/temperature
    # numerically stable log softmax
    shift_x = x - np.max(x)
    # mask invalid values (neg inf)
    b = (~np.ma.masked_invalid(shift_x).mask).astype(int)
    return shift_x - logsumexp(shift_x, b=b)

  
def binary_search(a, x): 
    i = bisect_left(a, x) 
    if i != len(a) and a[i] == x: 
        return i 
    else: 
        return -1

def perplexity(arr):
    if len(arr) == 0:
        return INF
    score = sum([s for s in arr])
    return 2**(-score/len(arr))


def prod(iterable):
    return reduce(operator.mul, iterable, 1.0)

# Functions for common access to numpy arrays, lists, and dicts
def common_viewkeys(obj):
    """Can be used to iterate over the keys or indices of a mapping.
    Works with numpy arrays, lists, and dicts. Code taken from
    http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python
    """
    if isinstance(obj, dict):
        return obj.keys()
    else:
        return range(len(obj))


def common_iterable(obj):
    """Can be used to iterate over the key-value pairs of a mapping.
    Works with numpy arrays, lists, and dicts. Code taken from
    http://stackoverflow.com/questions/12325608/iterate-over-a-dict-or-list-in-python
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value
    else:
        for index, value in enumerate(obj):
            yield index, value


def common_get(obj, key, default):
    """Can be used to access an element via the index or key.
    Works with numpy arrays, lists, and dicts.
    
    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
        ``default`` (object): Default return value if ``key`` not found
    
    Returns:
        ``obj[key]`` if ``key`` in ``obj``, otherwise ``default``
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return obj[key] if key < len(obj) else default


def common_contains(obj, key):
    """Checks the existence of a key or index in a mapping.
    Works with numpy arrays, lists, and dicts.
    
    Args:
        ``obj`` (list,array,dict):  Mapping
        ``key`` (int): Index or key of the element to retrieve
    
    Returns:
        ``True`` if ``key`` in ``obj``, otherwise ``False``
    """
    if isinstance(obj, dict):
        return key in obj
    else:
        return key < len(obj)


# String Kernel

# takes a dictionary and returns a new dictionary where the values are transposed
def transpose_all_values(dictionary):
    transposed_dict = {}
    for key in dictionary.keys():
        transposed_dict[key] = np.transpose(dictionary[key])
    return transposed_dict


# normalization as in this paper:
# https://pdfs.semanticscholar.org/07f9/4059372818242a2d09a42580a827d2c77f73.pdf
def normalization_for_dynamic_programming_approach(K, decay):
    normalized_K = {}
    for key in K.keys():
        normalized_K[key] = K[key] / (decay ** (2*key))
    return normalized_K


def find_computed_kernel(s, t, p, decay, string_kernel_previous, string_kernel_current):
    if (str(s), str(t), p, decay) in string_kernel_current:
        # result can be loaded from current time step
        _, K = string_kernel_current[(str(s), str(t), p, decay)]
        return K

    elif (str(t), str(s), p, decay) in string_kernel_current:
        # result can be loaded from current time step
        _, K = string_kernel_current[(str(t), str(s), p, decay)]
        return K

    elif (str(s), str(t), p, decay) in string_kernel_previous:
        # result can be loaded from previous time step
        # this can happen if s and t had already reached EOS in the previous time step
        S, K = string_kernel_previous[(str(s), str(t), p, decay)]
        string_kernel_current[(str(s), str(t), p, decay)] = (S, K)
        return K

    elif (str(t), str(s), p, decay) in string_kernel_previous:
        # result can be loaded from previous time step
        # this can happen if s and t had already reached EOS in the previous time step
        S, K = string_kernel_previous[(str(t), str(s), p, decay)]
        string_kernel_current[(str(s), str(t), p, decay)] = (S, K)
        return K

def compute_kernel(s, t, p, decay, val):
    # K stores the kernel results for the substring lengths 1...n
    K = {}

    # k_ stores intermediate results
    k_ = {}

    # S stores intermediate results
    S = {}

    if len(s) < 3 and len(t) < 3:
        # do not search for previous result

        K[1] = 0
        k_[1] = np.zeros((len(s)+1, len(t)+1))
        for i in range(1, len(s) + 1):
            for j in range(1, len(t) + 1):
                if s[i - 1] == t[j - 1]:
                    k_[1][i, j] = decay ** 2
                    K[1] = K[1] + k_[1][i, j]

        if p > 1:
            for l in range(2, p + 1):
                K[l] = 0
                S[l] = np.zeros((len(s) + 1, len(t) + 1), dtype=np.double)
                k_[l] = np.zeros((len(s) + 1, len(t) + 1))
                for i in range(1, len(s) + 1):
                    for j in range(1, len(t) + 1):
                        S[l][i, j] = k_[l - 1][i, j] + decay * S[l][i - 1, j] + decay * S[l][i, j - 1] - (decay ** 2) * S[l][
                            i - 1, j - 1]
                        if s[i - 1] == t[j - 1]:
                            k_[l][i, j] = (decay ** 2) * S[l][i - 1, j - 1]
                            K[l] = K[l] + k_[l][i, j]

    else:
        # can reuse results from previous time step

        if len(s) == len(t):
            previous_value = val
            if previous_value is None:
                raise Exception("previous result not found in string_kernel_previous even though it should be there! ")
            else:
                previous_S, previous_K = previous_value

                # load k_[1] and K[1] from previous result
                k_[1] = np.zeros((len(s) + 1, len(t) + 1))
                K[1] = previous_K[1]

                for i in range(1, len(s)):
                    for j in [len(t)]:
                        if s[i - 1] == t[j - 1]:
                            k_[1][i, j] = decay ** 2
                            K[1] = K[1] + k_[1][i, j]
                for j in range(1, len(t) + 1):
                    for i in [len(s)]:
                        if s[i - 1] == t[j - 1]:
                            k_[1][i, j] = decay ** 2
                            K[1] = K[1] + k_[1][i, j]

                for l in range(2, p+1):
                    # load previous results
                    K[l] = previous_K[l]
                    S[l] = np.zeros((len(s)+1, len(t)+1), dtype=np.double)
                    S[l][:-1, :-1] = previous_S[l]
                    k_[l] = np.zeros((len(s)+1, len(t)+1))

                    for i in range(1, len(s)):
                        for j in [len(t)]:
                            S[l][i, j] = k_[l-1][i, j] + decay * S[l][i-1, j] + decay * S[l][i, j-1] - (decay ** 2) * S[l][i-1, j-1]
                            if s[i-1] == t[j-1]:
                                k_[l][i, j] = (decay ** 2) * S[l][i-1, j-1]
                                K[l] = K[l] + k_[l][i, j]
                    for j in range(1, len(t) + 1):
                        for i in [len(s)]:
                            S[l][i, j] = k_[l - 1][i, j] + decay * S[l][i - 1, j] + decay * S[l][i, j - 1] \
                                         - (decay ** 2) * S[l][i - 1, j - 1]
                            if s[i - 1] == t[j - 1]:
                                k_[l][i, j] = (decay ** 2) * S[l][i - 1, j - 1]
                                K[l] = K[l] + k_[l][i, j]

        elif len(s) > len(t):

            previous_value = val
            if previous_value is None:
                raise Exception("previous result not found in string_kernel_previous even though it should be there! ")

            previous_S, previous_K = previous_value

            # load k_[1] and K[1] from previous result
            k_[1] = np.zeros((len(s) + 1, len(t) + 1))
            K[1] = previous_K[1]

            for j in range(1, len(t) + 1):
                for i in [len(s)]:
                    if s[i - 1] == t[j - 1]:
                        k_[1][i, j] = decay ** 2
                        K[1] = K[1] + k_[1][i, j]

            for l in range(2, p + 1):
                # load previous results
                K[l] = previous_K[l]
                S[l] = np.zeros((len(s) + 1, len(t) + 1), dtype=np.double)
                S[l][:-1, :] = previous_S[l]
                k_[l] = np.zeros((len(s) + 1, len(t) + 1))

                for j in range(1, len(t) + 1):
                    for i in [len(s)]:
                        S[l][i, j] = k_[l - 1][i, j] + decay * S[l][i - 1, j] + decay * S[l][i, j - 1] \
                                     - (decay ** 2) * S[l][i - 1, j - 1]
                        if s[i - 1] == t[j - 1]:
                            k_[l][i, j] = (decay ** 2) * S[l][i - 1, j - 1]
                            K[l] = K[l] + k_[l][i, j]

        else:
            # len(t) > len(s)
            previous_value = val
            if previous_value is None:
                raise Exception("previous result not found in string_kernel_previous even though it should be there! ")

            previous_S, previous_K = previous_value

            # load k_[1] and K[1] from previous result
            k_[1] = np.zeros((len(s) + 1, len(t) + 1))
            K[1] = previous_K[1]

            for i in range(1, len(s) + 1):
                for j in [len(t)]:
                    if s[i - 1] == t[j - 1]:
                        k_[1][i, j] = decay ** 2
                        K[1] = K[1] + k_[1][i, j]

            for l in range(2, p + 1):
                # load previous results
                K[l] = previous_K[l]
                S[l] = np.zeros((len(s) + 1, len(t) + 1), dtype=np.double)
                S[l][:, :-1] = previous_S[l]
                k_[l] = np.zeros((len(s) + 1, len(t) + 1))

                for i in range(1, len(s) + 1):
                    for j in [len(t)]:
                        S[l][i, j] = k_[l - 1][i, j] + decay * S[l][i - 1, j] + decay * S[l][i, j - 1] - (decay ** 2) * \
                                     S[l][i - 1, j - 1]
                        if s[i - 1] == t[j - 1]:
                            k_[l][i, j] = (decay ** 2) * S[l][i - 1, j - 1]
                            K[l] = K[l] + k_[l][i, j]

    return S, K


# dynamic programming approach from this paper:
# https://pdfs.semanticscholar.org/07f9/4059372818242a2d09a42580a827d2c77f73.pdf
# optimized to reuse previous results
def dynamic_programming_substring_kernel_k_efficient(s, t, p, decay, string_kernel_previous, string_kernel_current):

    if p < 1:
        print("the string kernel is only defined for positive values")
        exit(1)

    K = find_computed_kernel(s, t, p, decay, string_kernel_previous, string_kernel_current)
    if not K:
        if len(s) >= 3 or len(t) >= 3:
            # compute previous results
            if len(s) == len(t):
                val = string_kernel_previous[(str(s[:-1]), str(t[:-1]), p, decay)]
            elif len(s) < len(t):
                val = string_kernel_previous[(str(s), str(t[:-1]), p, decay)]
            else:
                val = string_kernel_previous[(str(s[:-1]), str(t), p, decay)]
        else:
            val = None
        result = compute_kernel(s=s, t=t, p=p, decay=decay, val=val)
        string_kernel_current[(str(s), str(t), p, decay)] = result
        S, K = result

    return K


# normalization like in the Lodhi et al. paper: K_norm(s,t) = K(s,t) / (sqrt(K(s,s) * K(t,t)))
def lodhi_normalization(kernel_values, s, t, p, decay, string_kernel_previous, string_kernel_current):
    kernel_values_ss = dynamic_programming_substring_kernel_k_efficient(s, s, p, decay, string_kernel_previous,
                                                                        string_kernel_current)
    kernel_values_tt = dynamic_programming_substring_kernel_k_efficient(t, t, p, decay, string_kernel_previous,
                                                                        string_kernel_current)
    results = {}
    for i in range(1, p+1):
        if kernel_values[i] == 0:
            if s == t:
                results[i] = 1
            else:
                results[i] = 0
        else:
            results[i] = kernel_values[i] / (np.sqrt(kernel_values_ss[i] * kernel_values_tt[i]))
    return results


def get_string_kernel_value_to_subtract_test(hypo_index, hypo_array, selected_indices, string_kernel_n,
                                             string_kernel_decay, string_kernel_previous, string_kernel_current,
                                             mean=False):
    """Used for testing the string kernel.
       Helper function for string_kernel_diversity. Uses dynamic_programming_substring_kernel_k_efficient, normalizes
       with Lodhi's normalization and uses the result of substring length string_kernel_n.
       Returns the string kernel values to compute the similarity score to subtract from the score of
       hypo_array[hypo_index] to obtain the augmented probability.

        Args:
            ``hypo_index`` (int):  Index of the hypothesis for which we want to compute the augmented probability
            ``hypo_array`` (list): List of all hypotheses
            ``selected_indices`` (list): List of the indices that have been selected already
            ``string_kernel_n`` (int): Parameter 'n' for string kernel, denoting length of the subsequences to consider
            ``string_kernel_decay`` (double): Parameter 'decay' for string kernel
            ``string_kernel_previous`` (dict): String kernel results from previous time step
            ``string_kernel_current`` (dict): String kernel results from current time step
            ``mean`` (bool): if True, uses np.mean instead of just the n-th result of string kernel


        Returns:
            String kernel values for values 1..n (can be used to subtract from the probability of the
            hypothesis hypo_array[hypo_index])
        """
    if len(selected_indices) == 0:
        return 0
    elif len(selected_indices) == 1:
        kernel_values = dynamic_programming_substring_kernel_k_efficient(s=hypo_array[hypo_index].trgt_sentence,
                                                                         t=hypo_array[
                                                                             selected_indices[0]].trgt_sentence,
                                                                         p=string_kernel_n, decay=string_kernel_decay,
                                                                         string_kernel_previous=string_kernel_previous,
                                                                         string_kernel_current=string_kernel_current)
        if mean:
            return np.mean(list(lodhi_normalization(kernel_values=kernel_values, s=hypo_array[hypo_index].trgt_sentence,
                                                    t=hypo_array[selected_indices[0]].trgt_sentence,
                                                    p=string_kernel_n, decay=string_kernel_decay,
                                                    string_kernel_previous=string_kernel_previous,
                                                    string_kernel_current=string_kernel_current).values()))
        else:
            return lodhi_normalization(kernel_values=kernel_values, s=hypo_array[hypo_index].trgt_sentence,
                                       t=hypo_array[selected_indices[0]].trgt_sentence,
                                       p=string_kernel_n, decay=string_kernel_decay,
                                       string_kernel_previous=string_kernel_previous,
                                       string_kernel_current=string_kernel_current)[string_kernel_n]
    else:
        # build matrix and take determinant
        indices_to_compare = selected_indices.copy()
        indices_to_compare.append(hypo_index)
        num_indices_to_compare = len(indices_to_compare)
        matrix = np.zeros((num_indices_to_compare, num_indices_to_compare))
        for i in range(num_indices_to_compare):
            for j in range(num_indices_to_compare):
                kernel_values = dynamic_programming_substring_kernel_k_efficient(
                    s=hypo_array[indices_to_compare[i]].trgt_sentence,
                    t=hypo_array[indices_to_compare[j]].trgt_sentence,
                    p=string_kernel_n, decay=string_kernel_decay,
                    string_kernel_previous=string_kernel_previous,
                    string_kernel_current=string_kernel_current)
                if mean:
                    matrix[i][j] = np.mean(list(lodhi_normalization(kernel_values=kernel_values,
                                                                    s=hypo_array[indices_to_compare[i]].trgt_sentence,
                                                                    t=hypo_array[indices_to_compare[j]].trgt_sentence,
                                                                    p=string_kernel_n, decay=string_kernel_decay,
                                                                    string_kernel_previous=string_kernel_previous,
                                                                    string_kernel_current=string_kernel_current).values()))
                else:
                    matrix[i][j] = lodhi_normalization(kernel_values=kernel_values,
                                                       s=hypo_array[indices_to_compare[i]].trgt_sentence,
                                                       t=hypo_array[indices_to_compare[j]].trgt_sentence,
                                                       p=string_kernel_n, decay=string_kernel_decay,
                                                       string_kernel_previous=string_kernel_previous,
                                                       string_kernel_current=string_kernel_current)[string_kernel_n]

        return np.linalg.det(matrix)


def get_string_kernel_value_to_subtract(hypo_index, hypo_array, selected_indices, string_kernel_n, string_kernel_decay,
                                        string_kernel_previous, string_kernel_current, normalize=True):
    """Helper function for string_kernel_diversity. Uses dynamic programmming approach.
       Returns the value to subtract from the score of hypo_array[hypo_index] to obtain the augmented probability.

        Args:
            ``hypo_index`` (int):  Index of the hypothesis for which we want to compute the augmented probability
            ``hypo_array`` (list): List of all hypotheses
            ``selected_indices`` (list): List of the indices that have been selected already
            ``string_kernel_n`` (int): Parameter 'n' for string kernel, denoting length of the subsequences to consider
            ``string_kernel_decay`` (double): Parameter 'decay' for string kernel
            ``string_kernel_previous`` (dict): String kernel results computed in previous time step
            ``string_kernel_current`` (dict): String kernel results computed in current time step

        Returns:
            Value to subtract from the probability of the hypothesis hypo_array[hypo_index]
        """
    if len(selected_indices) == 0:
        return 0
    elif len(selected_indices) == 1:
        kernel_values = dynamic_programming_substring_kernel_k_efficient(s=hypo_array[hypo_index].trgt_sentence,
                                                                t=hypo_array[selected_indices[0]].trgt_sentence,
                                                                p=string_kernel_n, decay=string_kernel_decay,
                                                                string_kernel_previous=string_kernel_previous,
                                                                string_kernel_current=string_kernel_current)
        if normalize:
            kernel_values = lodhi_normalization(kernel_values=kernel_values, s=hypo_array[hypo_index].trgt_sentence,
                                                t=hypo_array[selected_indices[0]].trgt_sentence,
                                                p=string_kernel_n, decay=string_kernel_decay,
                                                string_kernel_previous=string_kernel_previous,
                                                string_kernel_current=string_kernel_current)
        
        return np.mean(list(kernel_values.values()))
    else:
        # build matrix and take determinant
        indices_to_compare = selected_indices.copy()
        indices_to_compare.append(hypo_index)
        num_indices_to_compare = len(indices_to_compare)
        matrix = np.zeros((num_indices_to_compare, num_indices_to_compare))
        for i in range(num_indices_to_compare):
            for j in range( num_indices_to_compare):
                kernel_values = dynamic_programming_substring_kernel_k_efficient(s=hypo_array[indices_to_compare[i]].trgt_sentence,
                                                                                t=hypo_array[indices_to_compare[j]].trgt_sentence,
                                                                                p=string_kernel_n, decay=string_kernel_decay,
                                                                                string_kernel_previous=string_kernel_previous,
                                                                                string_kernel_current=string_kernel_current)
                if normalize:
                    kernel_values = lodhi_normalization(kernel_values=kernel_values,
                                                        s=hypo_array[indices_to_compare[i]].trgt_sentence,
                                                        t=hypo_array[indices_to_compare[j]].trgt_sentence,
                                                        p=string_kernel_n, decay=string_kernel_decay,
                                                        string_kernel_previous=string_kernel_previous,
                                                        string_kernel_current=string_kernel_current)

                matrix[i][j] = np.mean(list(kernel_values.values()))
                   
        return np.linalg.det(matrix)


def select_with_string_kernel_diversity(arr, scores, n, string_kernel_n, string_kernel_decay, string_kernel_weight,
                                        string_kernel_state, method="original"):
    """Get indices of the ``n`` hypotheses from ``arr`` with the maximum scores
    after augmenting the scores with the string kernel diversity. The
    parameter ``arr`` is a list of PartialHypothesis. The returned index set is
    not guaranteed to be sorted.

    Args:
        arr (list):  List of PartialHypothesis objects
        n  (int):  Number of values to retrieve
        string_kernel_n (int):  n for subsequence kernel, denotes the length
                                of the subsequences to consider
        string_kernel_decay (float): decay factor for the string kernel
        string_kernel_weight (float): how much weight the similarity penalty should have
        string_kernel_state (dict): previously computed string kernel results

    Returns:
        List of indices of the ``n`` best hypotheses in ``arr``,
        considering the score and the diversity computed with the string kernel
    """
    if len(arr) <= n:
        return range(len(arr)), string_kernel_state

    # set with the selected indices
    selected_indices = []

    # update string kernel result dicts
    string_kernel_previous = string_kernel_state
    string_kernel_current = {}

    while len(selected_indices) < n:
        augmented_probs = []
        for i in range(len(arr)):
            if i not in selected_indices:

                similarity_score = get_string_kernel_value_to_subtract(hypo_index=i, hypo_array=arr,
                                                                       selected_indices=selected_indices,
                                                                       string_kernel_n=string_kernel_n,
                                                                       string_kernel_decay=string_kernel_decay,
                                                                       string_kernel_previous=string_kernel_previous,
                                                                       string_kernel_current=string_kernel_current)
                if TEST:
                    lodhi_test_val = get_string_kernel_value_to_subtract_compare_with_lodhi_recursive_normalized(i, arr,
                                                                                                    selected_indices,
                                                                                                    string_kernel_n,
                                                                                                    string_kernel_decay)
                    dynamic_test_val_n = get_string_kernel_value_to_subtract_test(hypo_index=i, hypo_array=arr,
                                                                                selected_indices=selected_indices,
                                                                                string_kernel_n=string_kernel_n,
                                                                                string_kernel_decay=string_kernel_decay,
                                                                                string_kernel_previous=string_kernel_previous,
                                                                                string_kernel_current=string_kernel_current)

                    dynamic_test_val_mean = get_string_kernel_value_to_subtract_test(hypo_index=i, hypo_array=arr,
                                                                                selected_indices=selected_indices,
                                                                                string_kernel_n=string_kernel_n,
                                                                                string_kernel_decay=string_kernel_decay,
                                                                                string_kernel_previous=string_kernel_previous,
                                                                                string_kernel_current=string_kernel_current,
                                                                                mean=True)

                    if abs(dynamic_test_val_n - lodhi_test_val) > 0.00000000001:
                        print("")
                        print(lodhi_test_val)
                        print(dynamic_test_val_n)
                        print("Difference: ", abs(dynamic_test_val_n - lodhi_test_val))
                        assert(abs(dynamic_test_val_n - lodhi_test_val) < 0.00000000001)

                    if abs(dynamic_test_val_mean - similarity_score) > 0.00000000001:
                        print("")
                        print(similarity_score)
                        print(dynamic_test_val_mean)
                        print("Difference: ", abs(dynamic_test_val_mean - similarity_score))
                        assert(abs(dynamic_test_val_mean - similarity_score) < 0.00000000001)

                if method == "prob":
                    kernel_prob = string_kernel_weight*(1. - similarity_score)
                    hypotheses_score = log_add(scores[i], np.log(kernel_prob, where=kernel_prob!=0)) 
                elif method == "log":
                    hypotheses_score = scores[i] - string_kernel_weight * np.log(similarity_score, where=similarity_score!=0)
                else:
                    hypotheses_score = scores[i] - string_kernel_weight * similarity_score
                augmented_probs.append(hypotheses_score)
            else:
                # if index was already selected, give it negative infinity probability
                augmented_probs.append(-np.infty)
        selected_indices.append(np.argmax(augmented_probs))
    # return the n indices with the best augmented score
    return selected_indices, string_kernel_current


# String Kernel + DPP + Fast Greedy MAP Inference

def find_previous_val(s, t, string_kernel_n, string_kernel_decay, string_kernel_previous):
    len_s = len(s)
    len_t = len(t)

    # compute kernel
    if len_s >= 3 or len_t >= 3:
        # compute previous results
        if len_s == len_t:
            key = (str(s[:-1]), str(t[:-1]), string_kernel_n, string_kernel_decay)
            if key in string_kernel_previous:
                val = string_kernel_previous[key]
            else:
                # use symmetric result
                val = string_kernel_previous[(str(t[:-1]), str(s[:-1]), string_kernel_n, string_kernel_decay)]
                S = transpose_all_values(val[0])
                val = S, val[1]
        elif len_s < len_t:
            key = (str(s), str(t[:-1]), string_kernel_n, string_kernel_decay)
            if key in string_kernel_previous:
                val = string_kernel_previous[key]
            else:
                # use symmetric result
                val = string_kernel_previous[(str(t[:-1]), str(s), string_kernel_n, string_kernel_decay)]
                S = transpose_all_values(val[0])
                val = S, val[1]
        else:
            key = (str(s[:-1]), str(t), string_kernel_n, string_kernel_decay)
            if key in string_kernel_previous:
                val = string_kernel_previous[key]
            else:
                # use symmetric result
                val = string_kernel_previous[(str(t), str(s[:-1]), string_kernel_n, string_kernel_decay)]
                S = transpose_all_values(val[0])
                val = S, val[1]
    else:
        val = None
    return val


# computes p.s.d. L matrix
def compute_log_L(arr, scores, n, string_kernel_n, string_kernel_decay, string_kernel_weight, string_kernel_previous,
                  string_kernel_current, method1=True):
    num_hypotheses = len(arr)
    matrix = np.zeros((num_hypotheses, num_hypotheses))
    probs = scores - logsumexp(scores)


    start = time.time()
    process_pool = multiprocessing.Pool(4)
    unnormalized_results = {}
    indices_to_compute_K = []
    arguments_to_compute_K = []
    
    start = time.time()
    for i in range(num_hypotheses):
        for j in range(i + 1, num_hypotheses):
            s = arr[i].trgt_sentence
            t = arr[j].trgt_sentence
            # see if result has already been computed
            kernel_values = find_computed_kernel(s=s, t=t, p=string_kernel_n, decay=string_kernel_decay,
                                                 string_kernel_previous=string_kernel_previous,
                                                 string_kernel_current=string_kernel_current)
            if not kernel_values:
                # hasn't been computed yet. need to compute it
                indices_to_compute_K.append(((i,j), (s,t)))
                # get the results from the previous time step that should be reused
                val = find_previous_val(s, t, string_kernel_n, string_kernel_decay, string_kernel_previous)
                # prepare the arguments for multiprocessing
                arguments_to_compute_K.append((s, t, string_kernel_n, string_kernel_decay, val))
            else:
                # has already been computed
                unnormalized_results[(i,j)] = kernel_values
    print("preparation took: ", time.time()-start)

    start = time.time()
    # compute the kernel values
    results_array = process_pool.starmap(compute_kernel, arguments_to_compute_K)
    process_pool.close()
    process_pool.join()
    results_array = results_array
    print("multiprocessing took: ", time.time()-start)

    start = time.time()
    # update string_kernel_current
    for i in range(len(results_array)):
        string_kernel_current[(str(indices_to_compute_K[i][1][0]), str(indices_to_compute_K[i][1][1]), string_kernel_n, string_kernel_decay)] = results_array[i]
        unnormalized_results[indices_to_compute_K[i][0]] = results_array[i][1]
    print("update: ", time.time() - start)

    start = time.time()
    # normalize the kernel values
    for i in range(num_hypotheses):
        for j in range(i + 1, num_hypotheses):
            kernel_values = lodhi_normalization(kernel_values=unnormalized_results[(i, j)],
                                                s=arr[i].trgt_sentence,
                                                t=arr[j].trgt_sentence,
                                                p=string_kernel_n, decay=string_kernel_decay,
                                                string_kernel_previous=string_kernel_previous,
                                                string_kernel_current=string_kernel_current)

            matrix[i][j] = np.mean(list(kernel_values.values()))
    print("normalize: ", time.time() - start)
    matrix = matrix + np.transpose(matrix) + np.eye(matrix.shape[0])

    log_matrix = np.log(matrix*string_kernel_weight, where=matrix > 0)
    log_matrix[matrix == 0] = NEG_INF
    if TEST:
        assert (np.linalg.eigvals(matrix*string_kernel_weight) >= 0).all()

    if method1:
        diags = vec_log_add(np.log(string_kernel_weight), probs)
        np.fill_diagonal(log_matrix, diags)
        return log_matrix

    else:
        sq_probs = 0.5*probs
        return sq_probs+np.transpose(sq_probs+log_matrix)

# computes K-ensemble given L
def compute_K(L):
    dim_L = L.shape[0]
    inverse_term = np.linalg.inv(np.identity(dim_L)+L)
    return np.matmul(inverse_term, L)

# fast greedy map inference algorithm. returns list of n selected indices.
def fast_greedy_map_inference(L, n):

    # initialize
    k_2 = L.shape[0]
    c = [[] for i in range(k_2)]
    d_2 = [L[i, i] for i in range(k_2)]
    j = argmax([np.log(d) for d in d_2])
    Y_g = {j}

    # iterate
    while len(Y_g) < n:
        for i in range(k_2):
            if i not in Y_g:
                e_i = (L[j, i] - float(np.inner(c[j], c[i])))/np.sqrt(d_2[j])
                c[i].append(e_i)
                d_2[i] = d_2[i] - e_i ** 2
        log_d_2 = [np.log(d) for d in d_2]
        log_d_2_without_Y_g = [-np.inf if index in Y_g else log_d_2[index] for index in range(k_2)]
        j = argmax(log_d_2_without_Y_g)
        Y_g.add(j)

    assert(len(Y_g) == n)
    return list(Y_g)

# fast greedy map inference algorithm. returns list of n selected indices.
def log_fast_greedy_map_inference(log_L, n):

    # initialize
    k_2 = log_L.shape[0]
    c = [[] for i in range(k_2)]
    sign_bits = [[] for i in range(k_2)]
    log_d_2 = [log_L[i, i] for i in range(k_2)]
    j = argmax(log_d_2)
    Y_g = {j}

    # iterate
    while len(Y_g) < n:
        for i in range(k_2):
            if i not in Y_g:
                sign, log_inner = bit_logsumexp(c[i], c[j], sign_bits[i], sign_bits[j])
                func = log_add if sign == -1 else log_minus
                if log_L[j, i] > log_inner: 
                    sign = 1
                    e_i = func(log_L[j, i], log_inner) - 0.5*log_d_2[j]
                else:
                    sign = -sign
                    e_i = func(log_inner,log_L[j, i]) - 0.5*log_d_2[j]
                c[i].append(e_i)
                sign_bits[i].append(sign)
                log_d_2[i] = log_minus(log_d_2[i], 2*e_i)

        log_d_2_without_Y_g = [-np.inf if index in Y_g else log_d_2[index] for index in range(k_2)]
        j = argmax(log_d_2_without_Y_g)
        Y_g.add(j)

    assert(len(Y_g) == n)
    return list(Y_g)

def bit_logsumexp(arr1, arr2, sign_arr1, sign_arr2):
    sign, log_inner = 1, NEG_INF
    for ind, (x,y) in enumerate(zip(arr1, arr2)):
        cur_sign = sign_arr1[ind]*sign_arr2[ind]
        func = log_add if sign == cur_sign else log_minus
        if log_inner > x+y: 
            log_inner = func(log_inner, x+y)
        else:
            sign = cur_sign
            log_inner = func(x+y, log_inner)
    return sign, log_inner


def select_with_fast_greedy_map_inference(arr, scores, n, string_kernel_n, string_kernel_decay, 
                                          string_kernel_weight, string_kernel_state):
    """Get indices of the ``n`` hypotheses from ``arr`` selected with the fast greedy MAP inference algorithm.
        Uses string kernel. The parameter ``arr`` is a list of PartialHypothesis.

        Args:
            arr (list):  List of PartialHypothesis objects
            scores (list): List of scores of the arr objects
            n  (int):  Number of values to retrieve
            string_kernel_n (int):  n for subsequence kernel, denotes the length
                                    of the subsequences to consider
            string_kernel_decay (float): decay factor for the string kernel
            string_kernel_state (dict): previously computed string kernel results

        Returns:
            List of indices of the ``n`` best hypotheses in ``arr``,
            considering the score and the diversity computed with the string kernel,
            selected with the fast greedy MAP inference algorithm
        """
    assert string_kernel_weight >= 0
    if len(arr) <= n:
        return range(len(arr)), string_kernel_state

    string_kernel_previous = string_kernel_state
    string_kernel_current = {}
    log_L = compute_log_L(arr, scores, n, string_kernel_n, string_kernel_decay, string_kernel_weight,
        string_kernel_previous=string_kernel_previous, string_kernel_current=string_kernel_current)
    selected_indices = log_fast_greedy_map_inference(log_L, n)
    if TEST: 
        K = compute_K(np.exp(log_L))
        new_matrix = K[np.ix_(selected_indices, selected_indices)]
        print("New set probability:", np.linalg.det(new_matrix))
        print("Old set probability:", prod([K[i][i] for i in selected_indices]))
        print('--------')

    return selected_indices, string_kernel_current

# Miscellaneous

def get_path(tmpl, sub = 1):
    """Replaces the %d placeholder in ``tmpl`` with ``sub``. If ``tmpl``
    does not contain %d, return ``tmpl`` unmodified.
    
    Args:
        tmpl (string): Path, potentially with %d placeholder
        sub (int): Substitution for %d
    
    Returns:
        string. ``tmpl`` with %d replaced with ``sub`` if present
    """
    try:
        return tmpl % sub
    except TypeError:
        pass
    return tmpl


def split_comma(s, func=None):
    """Splits a string at commas and removes blanks."""
    if not s:
        return []
    parts = s.split(",")
    if func is None:
        return [el.strip() for el in parts]
    return [func(el.strip()) for el in parts]


def ngrams(sen, n):
    sen = sen.split(' ')
    output = []
    for i in range(len(sen)-n+1):
        output.append(tuple(sen[i:i+n]))
    return output

def distinct_ngrams(hypos, n):
    total_ngrams = 0
    distinct = []
    for h in hypos:
        all_ngrams = ngrams(h, n)
        total_ngrams += len(all_ngrams)
        distinct.extend(all_ngrams)
    
    if len(distinct) == 0:
        return 0
    return float(len(set(distinct)))/len(distinct)

def ngram_diversity(hypos):
    ds = [distinct_ngrams(hypos, i) for i in range(1,5)]
    return sum(ds)/4


MESSAGE_TYPE_DEFAULT = 1
"""Default message type for observer messages """


MESSAGE_TYPE_POSTERIOR = 2
"""This message is sent by the decoder after ``apply_predictors`` was
called. The message includes the new posterior distribution and the
score breakdown. 
"""


MESSAGE_TYPE_FULL_HYPO = 3
"""This message type is used by the decoder when a new complete 
hypothesis was found. Note that this is not necessarily the best hypo
so far, it is just the latest hypo found which ends with EOS.
"""


class Observer(object):
    """Super class for classes which observe (GoF design patten) other
    classes.
    """
    
    @abstractmethod
    def notify(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Get a notification from an observed object.
        
        Args:
            message (object): the message sent by observed object
            message_type (int): The type of the message. One of the
                                ``MESSAGE_TYPE_*`` variables
        """
        raise NotImplementedError
    

class Observable(object):
    """For the GoF design pattern observer """
    
    def __init__(self):
        """Initializes the list of observers with an empty list """
        self.observers = []
    
    def add_observer(self, observer):
        """Add a new observer which is notified when this class fires
        a notification
        
        Args:
            observer (Observer): the observer class to add
        """
        self.observers.append(observer)
    
    def notify_observers(self, message, message_type = MESSAGE_TYPE_DEFAULT):
        """Sends the given message to all registered observers.
        
        Args:
            message (object): The message to send
            message_type (int): The type of the message. One of the
                                ``MESSAGE_TYPE_*`` variables
        """
        for observer in self.observers:
            observer.notify(message, message_type)

