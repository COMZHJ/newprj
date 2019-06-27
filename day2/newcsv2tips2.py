import pandas as pd
import matplotlib.pyplot as plt
# 修改默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']

print('-'*100)
ss = pd.read_csv('../data/tips.csv')
ss.info()
ss.head()

print('就餐的日期(星期)和小费金额是否有一定的关联')
day_mean = ss.groupby('day')['tip'].mean()
plt.bar(day_mean.index, day_mean.values, color='#ff0000', alpha=0.6)
plt.title('就餐日期与小费柱状图')
plt.show()

print('就餐的日期(星期)和小费金额是否有一定的关联')
day_sum = ss.groupby('day')['tip'].sum()
plt.plot(day_sum.index, day_sum.values, marker='D', color='#FF0000', linestyle='--')
plt.show()

print('就餐类型所占的百分比')
time = ss['time'].value_counts()
plt.pie(labels=time.index, x=time.values, autopct='%4.2f%%')
plt.show()
