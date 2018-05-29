import numpy as np
import string
import os
import operator

from nltk import pos_tag, word_tokenize


def init_weight(Mi, Mo):
	# weight initialization for neural network
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def all_parity_pairs(nbit):
    # total number of samples (Ntotal) will be a multiple of 100
    # generate all possible combinations of bits
    N = 2**nbit
    remainder = 100 - (N % 100)
    Ntotal = N + remainder
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N
        # now generate the ith sample
        for j in range(nbit):
            if i % (2**(j+1)) != 0:
                i -= 2**j
                X[ii,j] = 1
        Y[ii] = X[ii].sum() % 2
    return X, Y


def remove_punctuation(s):
	return s.translate(None, string.punctuation)


