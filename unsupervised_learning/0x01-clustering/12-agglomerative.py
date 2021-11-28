#!/usr/bin/env python3
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt

def agglomerative(X, dist):
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, dist, 'distance')
    scipy.cluster.hierarchy.dendrogram(Z, labels=clss, color_threshold=dist)
    plt.show()
    return clss
