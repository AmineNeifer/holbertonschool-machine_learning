#!/usr/bin/env python3
import sklearn.mixture

def gmm(X, k):
    model = sklearn.mixture.GaussianMixture(k).fit(X)
    return model.weights_, model.means_, model.covariances_, model.predict(X), model.bic(X)
