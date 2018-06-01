import numpy as np


def init_weight(Mi, Mo):
	# weight initialization for neural network
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)
