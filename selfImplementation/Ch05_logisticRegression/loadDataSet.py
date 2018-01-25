# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 19:31:26 2017

@author: jercas
"""

def loadDataSet():
	"""
		load data from data set
		Args:

		Returns:
			dataSet: train input of x
			labelSet: train input of y
	"""
	# initialize x-trainInput,y-trainInput
	dataSet = []
	labelSet = []
	# open file reader
	fr = open('testSet.txt')
	for line in fr.readlines():
		# strip() -- get rid of the space on both side
		# split() -- division as tab
		lineArr = line.strip().split()
		# padding data in list
		# x0 = 1.0 , x1 = column1 , x2 = column2
		dataSet.append([1.0, float(lineArr[0]), float(lineArr[1])])
		# label = column3
		labelSet.append(float(lineArr[2]))

	return dataSet,labelSet