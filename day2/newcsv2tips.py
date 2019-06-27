import pandas as pd
import seaborn as sb

ss = sb.load_dataset('tips')
# ss.to_csv('../data/test.csv', index=False)
# ss.to_csv('../data/test.csv', index=True, index_label='row_id')
print(ss)

print('-'*100)
ss = pd.read_csv('../data/tips.csv')
ss.info()
ss.head()

print('小费金额与消费总金额是否存在相关性')
# 相关性一般都是散点图
total_bill = ss['total_bill']
tip = ss['tip']

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.scatter(total_bill, tip)
plt.xlabel('消费的总金额')
plt.ylabel('小费')
plt.title('消费与小费的散点图')
plt.show()

print('性别和小费金额是否有一定关联')
female_mean = ss[ss['sex'] == 'Female']['tip'].mean()
male_mean = ss[ss['sex'] == 'Male']['tip'].mean()

plt.bar(['female', 'male'], [female_mean, male_mean], width=0.5, color='#ff0000')
plt.title('性别与小费的柱状图')
plt.show()
