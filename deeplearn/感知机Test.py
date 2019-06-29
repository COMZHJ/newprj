# 对矩阵和张量处理的api
import numpy as np
import matplotlib.pyplot as plt

#  修改默认字体，否则会有中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
print('1: 随机生成一条直线')
# 默认生成50个等差数列
x = np.linspace(0, 1, 50)
# 随机产生一条线
a = np.random.rand()
b = np.random.rand()
print(f'a={a},b={b}')
# 生成直线的函数方程
f = lambda x: a * x + b
# 其实不一定要生成50个值，x生成2个值即可
plt.plot(x, f(x), 'r')

print('2: 前面的直线把生成的100个点分成了2类')
# 产生[100,2]个点 (特征值)
N = 100
xn = np.random.rand(N, 2)
# 用来存储每个点的目标值
yn = np.zeros([N, 1])
# 通过之前的线把生成的100个点随机分成了2类
for i in range(N):
    if (f(xn[i, 0]) >= xn[i, 1]):
        # 当前 x[i] 点的坐标在直线的下方
        yn[i] = -1
        # plot([x], y, [fmt], data=None, **kwargs)
        #  fmt = '[color][marker][line]'
        # 百度搜索关键字："plt.plot 参数"
        plt.plot(xn[i, 0], xn[i, 1], 'bx', markersize=8)
    else:
        # 当前 x[i] 点的坐标在直线的上方
        yn[i] = 1
        plt.plot(xn[i, 0], xn[i, 1], 'go', markersize=8)
plt.show()

print('3：创建感知机通过反向传播误差不断调整权重与偏置')
# 感知机的实现，对于给定的x,y值，和之前的标准答案，感知机要找到分类的超平面
def perceptron(xn, yn, MaxIter=1000, a=0.1, w=np.zeros(3)):
    '''
            实现一个二维感知机: 对于给定的(x,y)，感知机通过迭代寻找最佳超平面（超平面永远是当前维度减-1）
        :param xn: 数据点：N*2向量
        :param yn: 分类结果：N*1向量
        :param MaxIter:  最大迭代次数，可选参数
        :param a: 学习率 (可选参数): 更新权重的一个比例
        :param w: 初始值 (可选参数): 权重，初始值都为零
        :return: 超平面参数使得 y = ax + b 最好的分割平面

        注意：由于初始值为随机数，因此迭代到收敛可能需要一点时间
    '''
    N = xn.shape[0]
    # 构造一个函数，对数据点进行分类
    # np.sinn 激活函数，可以把结果变成1或者-1
    # w[0] * 1是偏置，x[0]是第一个特征值 x[1]是第二个特征值
    f = lambda x: np.sign(w[0] * 1 + w[1] * x[0] + w[2] * x[1])
    # 此部分是反向传播的过程
    for _ in range(MaxIter):
        # 如果N为100则在1~100当中随机取出一个数
        i = np.random.randint(N)
        # 如果分类没有完全准确，则进行调整 (反向传播)
        # yn[i] 第i个样本的目标值，而 xn[i,:] 则是特征值, f(xn[i,:])则是预测值
        if (yn[i] != f(xn[i, :])):   # yn[i] 目前里面就是 -1或者1
            # 如果有差异来更新权重：权重原值 + 差异值 * 输入值 * 学习率
            w[0] = w[0] + yn[i] * 1 * a
            w[1] = w[1] + yn[i] * xn[i, 0] * a
            w[2] = w[2] + yn[i] * xn[i, 1] * a
    return w

print('5：通过权重与偏置求出新的a与b的值')
# 代码的实际应用, 目前的xn,yn等同于训练集的特征值与目标值
w = perceptron(xn, yn, MaxIter=10000)
print(w)
# 利用权重值w 计算y =ax + b中的a和b值
bnew = -w[0] / w[2]
anew = -w[1] / w[2]

print('5：传入x将会生成预测的分类线')
y = lambda x: anew * x + bnew
# 分割颜色
sep_color = (yn) / 2.0
print('color',sep_color.flatten())
# c是颜色序列,不应该是一个单一的RGB因为不遍区分
plt.scatter(xn[:, 0], xn[:, 1], c=sep_color.flatten(), s=50)
plt.plot(x, y(x), 'b--', label='感知机分类结果')
plt.plot(x, f(x), 'r', label='原始分类曲线')
plt.legend()
plt.title('原始分类曲线与感知机近似结果比较')
plt.show()

