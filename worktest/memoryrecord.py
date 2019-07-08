import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# 在内存中生成模拟数据
def GenerateData(training_epochs, batchsize = 100):
    for i in range(training_epochs):
        train_X = np.linspace(-1, 1, batchsize)     # 生成-1~1之间的100个浮点数
        train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3   # y=2x，但是加入了噪声
        yield shuffle(train_X, train_Y), i     # 以生成器的方式返回


Xinput = tf.placeholder('float', (None))
Yinput = tf.placeholder('float', (None))

# 建立会话，获取并输出数据
training_epochs = 20
with tf.Session() as sess:
    for (x, y), ii in GenerateData(training_epochs):
        # 通过静态图（占位符）注入的方式传入数据
        xv, yv = sess.run([Xinput, Yinput], feed_dict={Xinput: x, Yinput: y})

        print(ii, '| x.shape:', np.shape(xv), '| x[:3]:', xv[:3])
        print(ii, '| y.shape:', np.shape(yv), '| y[:3]:', yv[:3])

# 显示模拟数据点
train_data = list(GenerateData(1))[0]
plt.plot(train_data[0][0], train_data[0][1], 'ro', label='Original data')
plt.legend()
plt.show()

