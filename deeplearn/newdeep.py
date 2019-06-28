import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# 修改默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']

lb = load_boston()
print(lb, type(lb))
X = lb['data']
y = lb.target
X = pd.DataFrame(data=X, columns=lb['feature_names'])

'''
def draw_scatter(X, y, xlb):
    plt.scatter(X, y)
    plt.xlabel(xlb)
    plt.ylabel('房价')
    plt.title(f'{xlb}与房价关系')
    plt.show()


draw_scatter(X['CRIM'], y, '犯罪率')
draw_scatter(X['RM'], y, '犯罪率')
draw_scatter(X['NOX'], y, '犯罪率')
draw_scatter(X['ZN'], y, '犯罪率')
'''

print('-'*100)
print(X.shape, type(X))
print(y.shape, type(y))
# 特征工程
# X = X.drop(['NOX', 'ZN'], axis=1)
# 此处正确率低是由于欠拟合， 通过多项式对特征值处理
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
print(X.shape, type(X))
X = pf.fit_transform(X)
print(X.shape, type(X))

print('-'*100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
# 如果有模型则直接使用
from sklearn.externals import joblib
try:
    lr = joblib.load('../data/lr.pkl')
    print('模型加载成功！')
except:
    lr = LinearRegression()
    # 训练模型
    lr.fit(X_train, y_train)
    print(f'获取模型训练的权重\n{lr.coef_}')
    joblib.dump(lr, '../data/lr.pkl')
# 通过测试集验证模型
y_predict = lr.predict(X_test)
# 根据均方误差求误差值
print('均方误差为：', mean_squared_error(y_test, y_predict))
# 正确率：测试集的特征值和目标值
print('正确率：', lr.score(X_test, y_test))

'''
x = [i for i in range(y_predict.shape[0])]
plt.plot(x[:30], y_predict[:30], color='#ff0000', marker='.', label='y_predict')
plt.plot(x[:30], y_test[:30], color='#00ff00', marker='.', label='y_test')
plt.plasma()
plt.show()
'''
