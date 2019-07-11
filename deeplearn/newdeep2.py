import tensorflow as tf
import numpy as np
# 张量：变量、常量、占位符

a = tf.placeholder(dtype=tf.int32, shape=[2, 3])
b = tf.placeholder(dtype=tf.int32, shape=[2, 3])
c = tf.multiply(a, b)   # a * b
d = tf.placeholder(dtype=tf.int32, shape=[3, 1])
e = tf.matmul(a, d)     # a.dot(d)

with tf.Session() as sess:
    temp = sess.run(c, feed_dict={a: np.arange(6).reshape(2, 3), b: np.arange(6).reshape(2, 3)})
    print(temp)
    temp = sess.run(e, feed_dict={a: np.arange(6).reshape(2, 3), d: np.arange(3).reshape(3, 1)})
    print(temp)

print('-'*100)
a = tf.Variable(initial_value=3.14, dtype=tf.float32, name='a')
# 缺省 dtype=tf.float32
b = tf.Variable(initial_value=3.14, name='b')
c = a + b
d = tf.Variable(initial_value=3.14, dtype=tf.float32, name='d')
e = tf.add(c, d, name='add_demo')

# 和数据库相同创建一个会话  sess -> 图
with tf.Session() as sess: # Session 会自动关闭
    tf.summary.FileWriter('../data/tftest/', sess.graph)
    # sess = tf.Session()
    # 所有变量在运行时才会赋值
    sess.run(tf.global_variables_initializer())
    # c 也是变量（value shape dtype name）
    print(sess.run(e))
    # sess.close()

