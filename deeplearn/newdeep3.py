import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 自动生成的特征值
X = np.linspace(-3, 3, 100, dtype=np.float32)
# 增加一些随机数（目标值）
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 100)
# plt.scatter(X, y)
# plt.show()

# tf 中训练就是在训练权重 + 偏置必须用变量存储
w = tf.Variable(initial_value=0.0, name='w')
# 常量：特征值、目标值……变量：权重、偏置……占位符：在运行用来接收实际值
b = tf.Variable(initial_value=0, name='b', dtype=tf.float32)
# 特征值 * 权重 + 偏置 --> 预测值
y_predict = tf.add(tf.multiply(X, w), b, name='multiply_add')
# 获取均方误差
loss = tf.reduce_mean(tf.square(y - y_predict), name='reduce_mean')
# 梯度下降来减少误差 loss 步长0.01
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, 1001):
        sess.run(train_op)
        if i % 10 == 0:
            # 获取Session的值，Session.run()，eval()
            print(f'第{i}次均方误差为{loss.eval()}，权重为{w.eval()}，偏置为{b.eval()}')

    # 可视化显示
    plt.scatter(X, y)
    plt.plot(X, sess.run(y_predict))
    plt.show()

