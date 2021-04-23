#!/usr/bin/env python3
import pandas as pd


data = {'First': [0, 0.5, 1, 1.5], 'Second': ['one', 'two', 'three', 'four']}
df = pd.DataFrame.from_dict(data, orient='index',
                            columns=['A', 'B', 'C', 'D'])
df = df.T
