#!/usr/bin/env python3


"""gmm using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """guassian mm"""
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
