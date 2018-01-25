# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:12:00 2017

@author: jercas
"""
import matplotlib.pyplot as plt
import numpy as np
from loadDataSet import loadDataSet

def plot(weights):
    """
        plot data figure
        Args:
            weights: optima weights/thetas
        Returns:
            
    """
    # get training data & labels
    dataSet, labelSet = loadDataSet()
    # transform list to np.array
    dataArr = np.array(dataSet)
    # get number of training examples
    n = np.shape(dataArr)[0]

    # class-1 training data plot point position list (x,y)
    xcord1 = []
    ycord1 = []
    # class-2 training data plot point position list (x,y)
    xcord2 = []
    ycord2 = []

    # according to different classes to classified point
    for i in range(n):
        if int(labelSet[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    # ploting
    # get figure object
    fig = plt.figure()
    # plot subplot, one line-one graph-fitst plot
    ax = fig.add_subplot(111)
    # plot different scatter in the figure
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # prepare fitting line points
    x = np.arange(-3.0, 3.0, 0.1)
    y = np.array((-weights[0] - weights[1]*x) / weights[2]).T

    # plot fitting line
    ax.plot(x, y)
    # set axis label
    plt.xlabel('X1')
    plt.ylabel('X2')
    # show plot
    plt.show()
