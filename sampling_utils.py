import numpy as np
import scipy.optimize as opt
import utils
from bisect import bisect
import logging

def gumbel_max_sample(x, seed=None):
    """
    x: log-probability distribution (unnormalized is ok) over discrete random variable
    """
    if seed is not None:
        np.random.seed(seed=seed)
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return np.nanargmax(x + z)

def exponential_sample(x, seed=None):
    """
    probability distribution over discrete random variable
    """
    if seed is not None:
        np.random.seed(seed=seed)
    E = -np.log(np.random.uniform(size=len(x)))
    E /= x
    return np.nanargmin(E)

def log_multinomial_sample(x, seed=None):
    """
    x: log-probability distribution (unnormalized is ok) over discrete random variable
    """
    if seed is not None:
        np.random.seed(seed=seed)
    x[np.where(np.isnan(x))] = utils.NEG_INF
    c = np.logaddexp.accumulate(x) 
    key = np.log(np.random.uniform())+c[-1]
    return bisect(c, key)

def sample_k_dpp(lambdas, k):
    if k >= len(lambdas):
        return range(len(lambdas))
    N = len(lambdas)
    E = elem_polynomials(lambdas, k)
   
    J = []
    for n in range(N,0,-1):
        u = np.random.uniform()
        thresh = lambdas[n-1] * E[k-1,n-1] / E[k, n]
        if u < thresh:
            J.append(n-1)
            k -= 1
            if k == 0:
                break
    return J

def elem_polynomials(lambdas, k):
    N = len(lambdas)
    E = np.full((k+1,N+1), 0.)
    E[0,:] = 1.                     # initialization
    for i in range(1, k+1):
        for n in range(1,N+1):
            E[i,n] = E[i,n-1] + lambdas[n-1] * E[i-1,n-1]
    return E

def log_elem_polynomials(log_lambdas, k):
    N = len(log_lambdas)
    E = np.full((k+1,N+1), utils.NEG_INF)
    E[0,:] = 0.                     # initialization
    for i in range(1, k+1):
        for n in range(1,N+1):
            interm = log_lambdas[n-1] + E[i-1,n-1]
            E[i,n] = utils.log_add(E[i,n-1], interm) 
    return E

def log_elem_polynomial_newton(log_lambdas, k):

    def log_power_sum(log_lambdas, k):
        return utils.logsumexp(log_lambdas*k)

    pks = [log_power_sum(log_lambdas, i) for i in range(1, k+1)]

    eks = np.full(k+1, utils.NEG_INF, dtype=np.float128)
    eks[0] = 0.
    # keep track of sign bit
    sign = [1] * (k+1)

    for i in range(1, k+1):
        for j in range(1, i+1):
            s2 = (-1)**(j+1)*sign[i-j]
            func = utils.log_add if sign[i] == s2 else utils.log_minus
            if eks[i] > eks[i-j] + pks[j-1]:
                val1, val2 = eks[i], eks[i-j] + pks[j-1]
            else:
                sign[i] = s2
                val1, val2 = eks[i-j] + pks[j-1], eks[i]
            eks[i] = func(val1, val2)
        
        eks[i] -= np.log(i)
    return eks[-1]

def expected_k(log_X):
    return np.exp(utils.logsumexp([min(0.,i) for i in log_X]))

def get_const(log_lambdas, desired_k):
    """
    program for finding constant that gives us inclusion probabilities summing to 'desired_k'
    """
    base_inc_probs = np.log(desired_k) + log_lambdas 
    remaining_prob = 1 - np.exp(utils.logsumexp(log_lambdas))
    c = desired_k/expected_k(base_inc_probs)
    start = c*desired_k
    results = opt.minimize(lambda x: (desired_k - (expected_k(log_lambdas + x) + desired_k*remaining_prob))**2, np.log(start))
    return np.exp(results.x[0])

