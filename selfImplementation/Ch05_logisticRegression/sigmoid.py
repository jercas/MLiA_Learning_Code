# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 21:29:26 2017

@author: jercas
"""
import numpy as np

def sigmoid(z):
	"""
		sigmoid function
		Args:
			z: input independent variable
		Returns:
			output after sigmoid function processed
	"""
	# sigmoid function: g(z)= 1/(1+e^(-z))
	return 1.0 / (1 + np.exp(-z))