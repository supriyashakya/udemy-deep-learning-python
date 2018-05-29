import numpy as np
import string
import os
import operator

from nltk import pos_tag, word_tokenize


def init_weight(Mi, Mo):
	# weight initialization for neural network
	return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def remove_punctuation(s):
	return s.translate(str.maketrans('','',string.punctuation))


def get_robert_frost():
	word2idx = {'START': 0, 'END': 1}
	current_idx = 2
	sentences = []
	for line in open('robert_frost.txt'):
		line = line.strip()
		if line:
			tokens = remove_punctuation(line.lower()).split()
			sentence = []
			for t in tokens:
				if t not in word2idx:
					word2idx[t] = current_idx
					current_idx += 1
				idx = word2idx[t]
				sentence.append(idx)
			sentences.append(sentence)
	return sentences, word2idx


def my_tokenizer(s):
    s = remove_punctuation(s)
    s = s.lower()
    return s.split()