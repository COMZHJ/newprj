import numpy as np

print('求权重公式')
# 创建三个样本，四个特征
t1 = np.arange(12).reshape(3, 4)
# 创建四个权重
t2 = np.arange(4).reshape(4, 1)
# 点积[3,4] [4,1] = [3,1]
t3 = t1.dot(t2)

print(t1)
print(t2)
print(t3)

print('-'*100)
print('求误差公式')
# 真实值
y_true = np.arange(10).reshape(10, 1)
# 预测值
y_redict = np.arange(10, 20).reshape(10, 1)
# 线性回归中有均方误差公式
t4 = y_redict - y_true
t5 = t4 * t4
print(f'误差值{t5.mean()},{t5.sum()/t5.size}')

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_true, y_redict))
print(mean_squared_error(y_redict, y_true))

