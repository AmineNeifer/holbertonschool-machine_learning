#!/usr/bin/env python3
import sklearn.cluster

def kmeans(X, k):
    kmeans = sklearn.cluster.KMeans(k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
