#!/usr/bin/env python3

import pandas as pd


def from_numpy(array):
    col = len(array[0])
    alphabet_list = []
    alphabet_list[:0] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    df = pd.DataFrame(array, columns=alphabet_list[:col])
    return df
