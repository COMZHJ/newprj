# One-Hot编码：可以把文字映射为数字的方式
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
ohe.fit([['男', '中国', '足球'],
['女', '美国', '篮球'],
['男', '日本', '羽毛球'],
['女', '中国', '乒乓球']]) # 这里一共有4个数据，3种特征
array = ohe.transform([['男', '美国', '乒乓球']]).toarray() # 这里使用一个新的数据来测试
print(array)
print(ohe.inverse_transform(array))

