import numpy as np
import pandas as pd

ss = pd.Series(data=[1, 2, 3], index=list('abc'), dtype=np.float16, name='test')
print('-'*100)
print(ss, type(ss))
print('-'*100)
print(ss.index, ss.values, type(ss.values))

df = pd.DataFrame(data=np.arange(12).reshape(3, 4), index=list('abc'), columns=list('xyzw'), dtype=np.float16)
print('-'*100)
print(df, type(df))
print(df.index, df.columns)
print(df.values, type(df.values))


