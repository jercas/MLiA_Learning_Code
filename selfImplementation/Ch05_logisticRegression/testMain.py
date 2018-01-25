# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:06:26 2017

@author: jercas
"""
from gradAscent import batchGradAscent,stochasticGradAscent0, stochasticGradAscent1
from plot import plot
from loadDataSet import loadDataSet

def main():
    dataSet, labelSet = loadDataSet()
    weights = batchGradAscent(dataSet, labelSet)
    plot(weights)
    
    weights = stochasticGradAscent0(dataSet, labelSet)
    plot(weights)

    weights = stochasticGradAscent1(dataSet, labelSet)
    plot(weights)


if __name__ == '__main__':
    main()