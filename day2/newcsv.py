import numpy as np
import pandas as pd

ss = pd.read_csv('../data/groupby.csv')
ss.info()
print(ss.head(n=3))
print(ss[['Name', 'Brand']], type(ss[['Name', 'Brand']]))
print(ss.values, type(ss.values))

