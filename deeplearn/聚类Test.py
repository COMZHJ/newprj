# 线性回归预测房价
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
#  修改默认字体，否则会有中文乱码问题
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 加载数据
df = pd.read_excel("../data/user.xls", index_col="编号")
df.info()
print(df, type(df))
#  2: 选择模型
km = KMeans(n_clusters=4, max_iter=500)
# 训练之前需要做特征工程(归一化)
km.fit(df)
# 3：显示聚类的结果
print('样本的类别数(与样本数相等)', km.labels_, len(km.labels_))
print('类别的中心点:\n', km.cluster_centers_)
# 4: 把类别追加到数据中 (km.labels_ 是ndarray类型,没有索引是不可能进行列追加)
df = pd.concat([df,pd.Series(data=km.labels_, index=df.index, name='标签')], axis=1)
df.to_excel("../data/user_type.xls")
# 通过报表显示分类结果
arr = ['red', 'green', 'blue', 'gray']
colors = [arr[i] for i in km.labels_]
plt.scatter(df['入网时间'],df['消费金额'], color=colors)
plt.xlabel('入网时间', fontsize=12)
plt.ylabel('消费金额', fontsize=12)
plt.savefig("user.png")
plt.show()
