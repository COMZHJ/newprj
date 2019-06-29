import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据集：csv、dict、excel
# 训练集和测试集（一般0.8和0.2），如果训练集又可以分为训练集和验证集，这就会使用交叉验证
# 交叉验证：解决正确率不稳定的问题
mns = input_data.read_data_sets('../data/input_data', one_hot=True)
print(mns)

print(mns.train.images.shape)
print(mns.train.labels.shape)
# 按序去除指定数量的数据
print(mns.train.next_batch(1))
# print(mns.train.images[0])
# print(mns.train.labels[0])

# 占位符
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
# 添加隐藏层，让简单神经网络，变为深度神经网络
# 假设第一层 500个神经元 [None, 784] * [784, 500] --> [None, 500]
W1 = tf.Variable(initial_value=tf.random_normal(shape=[784, 500]), dtype=tf.float32)
B1 = tf.Variable(initial_value=tf.random_normal(shape=[500]), dtype=tf.float32)
# L1就是当前隐藏层的输出结果 [None, 784] * [784, 500] --> [None, 500]
L1 = tf.add(tf.matmul(X, W1), B1)
# 想要提高DNN命中率，应该在隐藏层的输出值后，再追加激活函数（多项式？？曲线？？）
L1 = tf.nn.tanh(L1)

# 假设第二层 300个神经元 [None, 500] * [500, 300] --> [None, 300]
W2 = tf.Variable(initial_value=tf.random_normal(shape=[500, 300]), dtype=tf.float32)
B2 = tf.Variable(initial_value=tf.random_normal(shape=[300]), dtype=tf.float32)
# L2就是当前隐藏层的输出结果 [None, 500] * [500, 300] --> [None, 300]
L2 = tf.add(tf.matmul(L1, W2), B2)
# 想要提高DNN命中率，应该在隐藏层的输出值后，再追加激活函数（多项式？？曲线？？）
L2 = tf.nn.tanh(L2)

# 有多少个像素（特征值）就会有多少个特征
# 有多少根连接线则有多少个权重 784 * 10
# 训练的过程就是不断修改权重和偏置的过程
# X[None,300] * [300, 10] --> [None, 10]
weight = tf.Variable(initial_value=tf.random.normal(shape=[300, 10]), dtype=tf.float32)
# 偏置，有多少个神经元就会有多少个偏置
bias = tf.Variable(initial_value=tf.random.normal(shape=[10]), dtype=tf.float32)
# 根据公式 y_predice = X * w + b 生成预测值
# [None,300] * [300, 10] --> [None, 10]
y_predict = tf.add(tf.matmul(L2, weight), bias)
# # 根据预测值和真实值的比较求误差（分类），只适合线性回归
# loss = tf.reduce_mean(tf.square(y - y_predict), name='reduce_mean')
# 而分类y_predict应该先转化为概率，再求误差
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_predict))

# 采用梯度下降减少误差
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 结果转换为正确率
a, b = tf.argmax(y_predict, axis=1), tf.argmax(y, axis=1)
result = tf.equal(a, b)
result = tf.cast(result, dtype=tf.float32)
result = tf.reduce_mean(result)

# 机器学习和深度学习都有“回归”和“分类”算法
# 回归：预测值（一个样本一个预测值）与真实值的误差
# 分类：输出的不是一个标准值，输出的是属于类别的概率（目标值必须是One-Hot编码）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # # 获取50张图 ==》 [50, 784]
    # image_list = mns.train.images[0:50]
    # result = sess.run(y_predict, feed_dict={X: image_list})
    # print(result)
    for i in range(5000):
        # 如果依赖了占位符，则运算时必须指定
        X_train, y_train = mns.train.next_batch(55)
        d = {X: X_train, y: y_train}
        sess.run(train_op, feed_dict=d)
        print(f'第{i}次的误差为{sess.run(loss, feed_dict=d)}，正确率为:{sess.run(result, feed_dict=d)}')

