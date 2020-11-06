#!/usr/bin/env python3


""" agglomerative function"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ agglomerative function"""
    hier = scipy.cluster.hierarchy.linkage(X, 'ward')
    clss = scipy.cluster.hierarchy.fcluster(
        Z=hier, t=dist, criterion="distance")
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(Z=hier, color_threshold=dist)
    plt.show()
    return clss
