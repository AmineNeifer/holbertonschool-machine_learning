#!/usr/bin/env python3

""" contains func create_confusion_matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates confuion matrix"""
    classes = logits.shape[1]
    m = logits.shape[0]
    conf_matrix = np.zeros((classes, classes))
    for i in range(m):
        conf_matrix[np.where(labels[i, :] == 1),
                    np.where(logits[i, :] == 1)] += 1
    return conf_matrix
