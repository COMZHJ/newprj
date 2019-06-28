import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 自动生成的特征值
X = np.linspace(-3, 3, 100, dtype=np.float32)
# 增加一些随机数（目标值）
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 100)
# plt.scatter(X, y)
# plt.show()

# # tf 中训练就是在训练权重 + 偏置必须用变量存储
# w = tf.Variable(initial_value=0.0, name='w')
# # 常量：特征值、目标值……变量：权重、偏置……占位符：在运行用来接收实际值
# b = tf.Variable(initial_value=0, name='b', dtype=tf.float32)
# # 特征值 * 权重 + 偏置 --> 预测值
# y_predict = tf.add(tf.multiply(X, w), b, name='multiply_add')

# 多项式的线性回归 y_predict = X2 + X3 + X * w + b
w1 = tf.Variable(initial_value=0.0, name='w1')
w2 = tf.Variable(initial_value=0.0, name='w2')
w3 = tf.Variable(initial_value=0.0, name='w3')
# w4 = tf.Variable(initial_value=0.0, name='w4')
# 常量：特征值、目标值……变量：权重、偏置……占位符：在运行用来接收实际值
b = tf.Variable(initial_value=0, name='b', dtype=tf.float32)
# 多项式的线性回归 y_predict = (X * w1 + b) + (X**2 * w2) + (X**3 * w3)
y_predict = tf.add(tf.multiply(X, w1), b)
y_predict = tf.add(tf.multiply(tf.pow(X, 2), w2), y_predict)
y_predict = tf.add(tf.multiply(tf.pow(X, 3), w3), y_predict, name='multiply_add')
# y_predict = tf.add(tf.multiply(tf.pow(X, 4), w4), y_predict, name='multiply_add')

# 获取均方误差
loss = tf.reduce_mean(tf.square(y - y_predict), name='reduce_mean')
# 可以把误差添加到图中
tf.summary.scalar('abc', loss)
# 把所有监控的Tensor整合Merge
merge = tf.summary.merge_all()

# 梯度下降来减少误差 loss 步长0.01
train_op = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

with tf.Session() as sess:
    fw = tf.summary.FileWriter('../data/tftest/', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(1, 1001):
        sess.run(train_op)
        fw.add_summary(merge.eval(), i)
        if i % 10 == 0:
            # 获取Session的值，Session.run()，eval()
            # print(f'第{i}次均方误差为{loss.eval()}，权重为{w.eval()}，偏置为{b.eval()}')
            # 多项式的线性回归 y_predict = X2 + X3 + X * w + b
            print(f'第{i}次均方误差为{loss.eval()}，偏置为{b.eval()}')

    # 可视化显示
    plt.scatter(X, y)
    plt.plot(X, sess.run(y_predict))
    plt.show()

