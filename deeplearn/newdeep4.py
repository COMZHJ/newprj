import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mns = input_data.read_data_sets('../data/input_data')
print(mns)

print(mns.train.images.shape)
print(mns.train.labels.shape)
# print(mns.train.images[0])
# print(mns.train.labels[0])

# 占位符
X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# 有多少个像素（特征值）就会有多少个特征
# 有多少根连接线则有多少个权重 784 * 10
# 训练的过程就是不断修改权重和偏置的过程
# X[None,785] * [784, 10] --> [None, 10]
weight = tf.Variable(initial_value=tf.random.normal(shape=[784, 10]), dtype=tf.float32)
# 偏置，有多少个神经元就会有多少个偏置
bias = tf.Variable(initial_value=tf.random.normal(shape=[10]), dtype=tf.float32)
# 根据公式 y_predice = X * w + b 生成预测值
# [50,785] * [784, 10] + [10] --> [50, 10]
y_predict = tf.add(tf.matmul(X, weight), bias)
# 根据预测值和真实值的比较求误差
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 获取20张图 ==》 [50, 784]
    image_list = mns.train.images[0:50]
    result = sess.run(y_predict, feed_dict={X: image_list})

    print(result)


