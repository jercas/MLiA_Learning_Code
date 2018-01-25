# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 21:07:00 2017

@author: jercas
"""
import numpy as np
import random
from sigmoid import sigmoid

def batchGradAscent(dataMatIn, labelMatIn):
	"""
		batch gradient ascent algorithm (traversal all of the training data set in one iteration) 
		-- find the optima of weights/thetas
		Args:
			dataMatIn: training/input data
			labelMatIn: label/actual result/output data
		Returns:
			weights: optima weights/thetas
	"""
	# transform train data list into matrix
	dataMatrix = np.mat(dataMatIn)
	# transform label data list into matrix
	labelVector = np.mat(labelMatIn).transpose()

	# m: number of input examples
	# n: number of features
	m,n = np.shape(dataMatrix)

	# parameters setting
	# learning rata
	alpha = 0.001
	# maximum iterations
	maxCycles = 500
	# weights / thetas
	weights = np.ones((n,1))

	# start gradent ascent
	for i in range(maxCycles):
		# hypothesis function: h(θ)=g(Xθ)
		h = sigmoid(dataMatrix * weights)
		# cost function: J(θ)=
		error = labelVector - h
		# gradient ascent: θ = θ + α*xT*(g(xθ)-y)/m
		#                     = θ + α*xT*(h-y)/m
		#                     = θ + α*xT*error/m
		weights = weights + alpha * dataMatrix.transpose() * error

	return weights


def stochasticGradAscent0(dataMatIn, labelMatIn):
	"""
		stochastic gradient ascent algorithm (traversal some of the training data set in one iteration) 
		-- find the optima of weights/thetas
		Args:
			dataMatIn: training/input data
			labelMatIn: label/actual result/output data
		Returns:
			weights: optima weights/thetas
	"""
	# transform data type for calculate
	dataMatIn = np.array(dataMatIn,dtype='float64')
	labelMatIn = np.array(labelMatIn,dtype='float64')

	m,n = np.shape(dataMatIn)
	alpha = 0.01
	weights = np.ones(n)

	# different from the BGA, SGA algorithm don't have matrix compution instead of scalar compution
	for i in range(m):
		h = sigmoid(sum(dataMatIn[i] * weights))
		error = labelMatIn[i] - h
		weights = weights + alpha * error * dataMatIn[i]

	return weights


def stochasticGradAscent1(dataMatIn, labelMatIn, numIter=150):
	"""
		improvement stochastic gradient ascent algorithm (traversal some of the training data set in one iteration)
		with dynamic learning rate alpha
		-- find the optima of weights/thetas
		Args:
			dataMatIn: training/input data
			labelMatIn: label/actual result/output data
		  numIter: training iteration
		Returns:
			weights: optima weights/thetas
	"""
	dataMatIn = np.array(dataMatIn,dtype='float64')
	labelMatIn = np.array(labelMatIn,dtype='float64')

	m,n = np.shape(dataMatIn)
	weights = np.ones(n)

	# start gradent ascent
	for j in range(numIter):
		# update all of the features
		dataIndex = range(m)
		for i in range(m):
			# dynamic change alpha value every iteration, at least = 0.01
			alpha = 4/(1.0 + j + i) + 0.01
			# change stochastic step
			randIndex = int(random.uniform(0, len(dataIndex)))

			# here we changed all the i to randIndex
			h = sigmoid(sum(dataMatIn[randIndex] * weights))
			error = labelMatIn[randIndex] - h
			weights = weights + alpha * error * dataMatIn[randIndex]

	return weights