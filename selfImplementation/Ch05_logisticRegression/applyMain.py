# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:11:26 2017

@author: jercas
"""
import numpy as np
from sigmoid import sigmoid
from gradAscent import stochasticGradAscent1

def classifyVector(trainingX, weights):
	"""
		distinguish different predict label by hypothesis output
		Args:
			trainingX: training data set
			weights: optima weights/thetas
		Returns:
			0: positive output
			1: negative output
	"""
	# hypothesis function: h(θ)=g(Xθ)
	prob = sigmoid(sum(trainingX * weights))

	if prob > 0.5:
		return 1.0
	else:
		return 0.0


def colicTest():
	"""
		open file and process data, then calculate predict error rate
		Args:
		Returns:
			errorRate: learning model predict error rate
	"""
	# get data set, compose input data set
	trainData = open('horseColicTraining.txt')
	testData = open('horseColicTest.txt')
	trainingSet = []
	trainingLabels = []

	# data preprocessing
	# traverse all of the training examples
	for line in trainData.readlines():
		# process data format
		currLine = line.strip().split('\t')
		lineArr = []
		# traverse all of the features in one examples
		for i in range(21):
			lineArr.append(float(currLine[i]))
		# columns 0-20 : features
		trainingSet.append(lineArr)
		# column  21 : label
		trainingLabels.append(float(currLine[21]))

	# training phase
	# data process over, then training model get the optimum
	trainWeights = stochasticGradAscent1(np.array(trainingSet), trainingLabels, 500)

	# error counter , testing data amount
	errorCount = 0
	numTest = 0.0

	# Testing phase
	for line in testData.readlines():
		numTest += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		# testing input data
		for i in range(21):
			lineArr.append(float(currLine[i]))

		# error judgement
		if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
			# predict error! , Count plus one
			errorCount += 1

	# calculate error rate
	errorRate = (float(errorCount) / numTest)
	print('The error rate of this test is:{0}'.format(errorRate))
	return errorRate


def multiTest():
	"""
		calculate average error rate
	"""
	numTests = 10
	errorSum = 0.0

	for i in range(numTests):
		errorSum += colicTest()

	print('after {0} iterations the average error rate is: {1}'.format(numTests, errorSum/float(numTests)))


def main():
	multiTest()


if __name__ == '__main__':
	main()