#!/usr/bin/env python3

"""
Contains class RNNCell that represents a cell of a simple RNN
"""
import numpy as np


class RNNCell:
	def __init_(self, i, h, o):
		"""
		Class constructor:

		@i is the dimensionality of the data
		@h is the dimensionality of the hidden state
		@o is the dimensionality of the outputs
		- Creates the public instance attributes @Wh, @Wy, @bh, @by
			@Wh and @bh are for the concatenated hidden state and input data
			@Wy and @by are for the output
		"""
		self.Wh = np.random.normal(-np.sqrt(1./i), np.sqrt(1./i), (h, i))
		self.bh = 0
		self.Wy = np.random.normal(-np.sqrt(1./i), np.sqrt(1./i), (h, i))
		self.by = 0
