#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(["Weighted_Price"], axis=1, inplace=True)
df["Volume_(BTC)"].fillna(value=0, inplace=True)
df["Volume_(Currency)"].fillna(value=0, inplace=True)
df["High"].fillna(method="ffill", inplace=True)
df["Low"].fillna(method="ffill", inplace=True)
df["Open"].fillna(method="ffill", inplace=True)
df["Close"].fillna(method="ffill", inplace=True)
df.fillna(method='ffill')
print(df.head())
print(df.tail())