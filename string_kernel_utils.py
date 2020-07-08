import itertools
import numpy as np

# dict that stores already computed values of the string kernel. format: ("k" or "k_prime", n, s, t, decay)
substring_kernel_results = {}


# helper function computing the summation in K', used in substring_kernel_k_prime()
def sum_for_computation_of_k_prime(j_indices, i, s, t, decay):
    sum_k_prime = 0
    for j in j_indices:
        sum_k_prime += substring_kernel_k_prime(s, t[:j-1], i-1, decay) * decay ** (len(t)-j+2)
    return sum_k_prime


# recursive computation of K'
def substring_kernel_k_prime(s, t, i, decay):
    key = ("k_prime", i, str(s), str(t), decay)
    symmetric_key = ("k_prime", i, str(t), str(s), decay)
    if i == 0:
        result = 1
        return result

    s_length = len(s)
    t_length = len(t)
    if min(s_length, t_length) < i:
        return 0

    if key in substring_kernel_results:
        # result has already been computed
        return substring_kernel_results[key]
    elif symmetric_key in substring_kernel_results:
        # symmetric result has already been computed (K(s,t)==K(t,s))
        return substring_kernel_results[symmetric_key]
    else:
        # result hasn't been computed yet, needs to be computed
        last_character_of_s = s[-1]
        # j+1 to have 1-indexing (to be consistent with the Lodhi et al. paper)
        j_where_tj_equals_x = [j+1 for j in range(t_length) if t[j] == last_character_of_s]
        result = decay * substring_kernel_k_prime(s[:-1], t, i, decay) + sum_for_computation_of_k_prime(j_where_tj_equals_x, i, s[:-1], t, decay)
        substring_kernel_results[key] = result
        return result


# helper function computing the summation in K, used in substring_kernel_k()
def sum_for_computation_of_k(j_indices, n, s, t, decay):
    sum_k = 0
    for j in j_indices:
        sum_k = sum_k + substring_kernel_k_prime(s, t[:j-1], n-1, decay) * decay ** 2
    return sum_k


# recursive computation of K
def substring_kernel_k(s, t, n, decay):
    key = ("k", n, str(s), str(t), decay)
    symmetric_key = ("k", n, str(t), str(s), decay)
    s_length = len(s)
    t_length = len(t)
    if min(s_length, t_length) < n:
        return 0

    if key in substring_kernel_results:
        # result has already been computed
        return substring_kernel_results[key]
    elif symmetric_key in substring_kernel_results:
        # symmetric result has already been computed (K(s,t)==K(t,s))
        return substring_kernel_results[symmetric_key]
    else:
        # result hasn't been computed yet, needs to be computed
        last_character_of_s = s[-1]
        # j+1 to have 1-indexing like in the Lodhi et al. paper
        j_where_tj_equals_x = [j+1 for j in range(len(t)) if t[j] == last_character_of_s]
        result = substring_kernel_k(s[:-1], t, n, decay) + sum_for_computation_of_k(j_where_tj_equals_x, n, s[:-1], t, decay)
        substring_kernel_results[key] = result
        return result


# given a string and an int n, returns a tuple (ngram_set, ngram_dict), where:
# ngram_set: set containing all (not necessarily contiguous) subsequences of length n from the string
# ngram_dict: dict, key is a subsequence and the value is a list of tuples, where each tuple contains the indices of
# the position of the subsequence + 1. example: string "cat", n=2, then for example ngram_dict["ct"] = [(1,3)]
def get_all_subsequences_of_length_n_from_string(string, n):
    # first, split string into characters
    string = list(string)
    # get all combinations of indices of length n
    combinations = list(itertools.combinations(range(len(string)), n))
    # build the n-grams and add them to the ngram_set
    # also, keep a ngram_dict that maps n-gram to a list of tuples of indices where it occurs
    ngram_set = set()
    ngram_dict = {}
    for combi in combinations:
        ngram = ""
        for i in range(n):
            ngram = ngram + string[combi[i]]
        ngram_set.add(ngram)
        if ngram in ngram_dict:
            # add indices of this ngram to the list of indices of this ngram
            # i+1 for 1-indexing (to be consistent with the Lodhi et al. paper)
            ngram_dict[ngram].append([i+1 for i in list(combi)])
        else:
            # create list of indices for this ngram and add the current indices
            # i+1 for 1-indexing (to be consistent with the Lodhi et al. paper)
            ngram_dict[ngram] = [[i+1 for i in list(combi)]]
    return ngram_set, ngram_dict


# this is the non-recursive definition of K' in the paper. can be used to check the recursive approach.
def test_substring_kernel_k_prime(s, t, n, decay):
    s_ngrams, s_ngram_dict = get_all_subsequences_of_length_n_from_string(s, n)
    t_ngrams, t_ngram_dict = get_all_subsequences_of_length_n_from_string(t, n)
    common_ngrams = s_ngrams.intersection(t_ngrams)
    result = 0
    for ngram in common_ngrams:
        for s_indices in s_ngram_dict[ngram]:
            for t_indices in t_ngram_dict[ngram]:
                result += decay ** (len(s) + len(t) - s_indices[0] - t_indices[0] + 2)
    return result


# this is the non-recursive definition of K in the paper. can be used to check the recursive approach.
def test_substring_kernel_k(s, t, n, decay):
    s_ngrams, s_ngram_dict = get_all_subsequences_of_length_n_from_string(s, n)
    t_ngrams, t_ngram_dict = get_all_subsequences_of_length_n_from_string(t, n)
    common_ngrams = s_ngrams.intersection(t_ngrams)
    result = 0
    for ngram in common_ngrams:
        for s_indices in s_ngram_dict[ngram]:
            for t_indices in t_ngram_dict[ngram]:
                result += decay ** ((s_indices[-1]-s_indices[0]+1)+(t_indices[-1]-t_indices[0]+1))
    return result


# recursive computation of K, normalized: K_norm(s,t) = K(s,t)/(sqrt(K(s,s)*K(t,t)))
def normalized_substring_kernel_k(s, t, n, decay):
    kernel_value = substring_kernel_k(s, t, n, decay)
    if kernel_value == 0:
        return 0
    else:
        normalization_term = np.sqrt(substring_kernel_k(s, s, n, decay) * substring_kernel_k(t, t, n, decay))
        return kernel_value / normalization_term
