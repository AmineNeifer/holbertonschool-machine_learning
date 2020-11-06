#!/usr/bin/env python3


"""kmean but with sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """k-means"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
