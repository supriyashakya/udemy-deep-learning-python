# class for gated recurrent unit

import numpy as np
import theano
import theano.tensor as tensor

from util import init_weight


class GRU:

	def __init__(self, Mi, Mo, activation):
		self.Mi = Mi
		self.Mo = Mo
		self.f = activation
		
		Wxr = init_weight(Mi, Mo) # input to r(t)
		Whr = init_weight(Mo, Mo) # hidden to r(t)
		br = np.zeros(Mo) # bias for r(t)
		Wxz = init_weight(Mi, Mo) # input to update
		Whz = init_weight(Mo, Mo) # hidden to update
		bz = np.zeros(Mo) # bias for update
		Wxh = init_weight(Mi, Mo) # input to hidden 
		Whh = init_weight(Mo, Mo) # hidden to hidden
		bh = np.zeros(Mo) # bias for hidden
		h0 = np.zeros(Mo) # initial hidden state

		self.Wxr = theano.shared(Wxr)
		self.Whr = theano.shared(Whr)
		self.br = theano.shared(br)
		self.Wxz = theano.shared(Wxz)
		self.Whz = theano.shared(Whz)
		self.bz = theano.shared(bz)
		self.Wxh = theano.shared(Wxh)
		self.Whh = theano.shared(Whh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(b0)
		self.params = [self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz, self.Wxh, self.Whh, self.bh, self.h0]

	def recurrence(self, x_t, h_t1):
		r = T.nnet.sigmoid(x_t.dot(self.Wxr) + h_t1.dot(self.Whr) + self.br)
		z = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
		hhat = self.f(x_t.dot(self.Wxh) + (r * h_t).dot(self.Whh) + self.bh)
		h = (1 - z) * h_t1 + z * hhat
		return h

	def output(self, x):
		h, _ = theano.scan(
			fn = self.recurrence,
			sequences = x,
			outputs_info = [self.h0],
			n_steps = x.shape[0]
		)
		return h
