from __future__ import division
import os
import sys
import numpy as np
from math import ceil, log, sqrt
from collections import Counter, defaultdict
import itertools
from sklearn.preprocessing import LabelEncoder

def log2(n):
    return log(n or 1, 2)

# ref: https://github.molgen.mpg.de/EDA/cisc/blob/master/formatter.py
def stratify(X, Y):
    """Stratifies Y based on unique values of X.
    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
    Returns:
        (dict): list of Y-values for a X-value
    """
    Y_grps = defaultdict(list)
    for i, x in enumerate(X):
        Y_grps[x].append(Y[i])
    return Y_grps

def map_to_majority(X, Y):
    """Creates a function that maps x to most frequent y.
    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
    Returns:
        (dict): map from Y-values to frequently co-occuring X-values
    """
    f = dict()
    Y_grps = stratify(X, Y)
    for x, Ys in Y_grps.items():
        frequent_y, _ = Counter(Ys).most_common(1)[0]
        f[x] = frequent_y
    return f

def update_regression(C, E, f, max_niterations=100):
    """Update discrete regression with C as a cause variable and Y as a effect variable
    so that it maximize likelihood
    Args
    -------
        C (sequence): sequence of discrete outcomes
        E (sequence): sequence of discrete outcomes
        f (dict): map from C to Y
    """
    supp_C = list(set(C))
    supp_E = list(set(E))
    mod_E = len(supp_E)
    n = len(C)

    # N_E's log likelihood
    # optimize f to minimize N_E's log likelihood
    cur_likelihood = 0
    res = [(e - f[c]) % mod_E for c, e in zip(C, E)]
    for freq in Counter(res).values():
        cur_likelihood += freq * (log2(n) - log2(freq))

    j = 0
    minimized = True
    while j < max_niterations and minimized:
        minimized = False

        for c_to_map in supp_C:
            best_likelihood = sys.float_info.max
            best_e = None

            for cand_e in supp_E:
                if cand_e == f[c_to_map]:
                    continue

                f_ = f.copy()
                f_[c_to_map] = cand_e


                neglikelihood = 0
                res = [(e - f_[c]) % mod_E for c, e in zip(C, E)]
                for freq in Counter(res).values():
                    neglikelihood += freq * (log2(n) - log2(freq))

                if neglikelihood < best_likelihood:
                    best_likelihood = neglikelihood
                    best_e = cand_e

            if best_likelihood < cur_likelihood:
                cur_likelihood = best_likelihood
                f[c_to_map] = best_e
                minimized = True
        j += 1

    return f

def cause_effect_negloglikelihood(C, E, func):
    """Compute negative log likelihood for effect given finction func: C -> E
    Model type : C→E
    Args
    -------
        C (sequence): sequence of discrete outcomes (Cause)
        E (sequence): sequence of discrete outcomes (Effect)
        func (dict): map from C-value to E-value
    Returns
    -------
        (float): maximum log likelihood
    """
    mod_C = len(set(C))
    mod_E = len(set(E))
    supp_C = list(set(C))
    supp_E = list(set(E))

    C_freqs = Counter(C)
    n = len(C)

    pair_cnt = defaultdict(lambda: defaultdict(int))
    for c, e in zip(C, E):
        pair_cnt[c][e] += 1

    loglikelihood = 0


    for e_E in supp_E:
        freq = 0
        for e in supp_E:
            for c in supp_C:
                if (func[c] + e_E) % mod_E == e:
                    freq += pair_cnt[c][e]
        loglikelihood += freq * (log2(n) - log2(freq))

    return loglikelihood
