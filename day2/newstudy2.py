import pandas as pd

df = pd.read_csv('../data/house.csv')
df.info()

X = df.drop(['price', 'row_id'], axis=1)
y_true = df['price']

print(X)
print('-'*100)
print(y_true)

print('-'*100)
from sklearn.model_selection import train_test_split
'''
训练特征值、测试特征值、训练目标值、测试目标值
random_state=1,2,3,4,5,6......：固定训练集和测试级
'''
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=3)
print(f'测试集数据\n{X_test}\n{y_test}')

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# 有目标值的就是有监督的机器学习
lr.fit(X_train, y_train)
print(f'获取模型训练的权重\n{lr.coef_}')
# 通过测试集验证模型
y_predict = lr.predict(X_test)
# 根据均方误差求误差值
from sklearn.metrics import mean_squared_error
print('均方误差为：', mean_squared_error(y_test, y_predict))

