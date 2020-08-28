#!/usr/bin/env python3


""" Contains early_stopping function"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ early stopping techniques to reduce overfitting"""
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0
    if count >= patience:
        return True, count
    return False, count
