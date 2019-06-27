import pandas as pd

ss = pd.read_csv('../data/groupby.csv')
ss.info()
print(ss.head(n=3))
print(ss[['Name', 'Brand']], type(ss[['Name', 'Brand']]))
print(ss.values, type(ss.values))

print('-'*100)
print(ss.sort_values('Count', ascending=False))

print('-'*100)
gp = ss.groupby('Brand')
for index, val in gp:
    print(index)
    print(val)
print(ss.groupby('Brand').count())
print(ss.groupby(by='Brand')['Count'].sum())

