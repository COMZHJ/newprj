import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mns = input_data.read_data_sets('../data/input_data')
print(mns)

print(mns.train.images.shape)
print(mns.train.labels.shape)
print(mns.train.images[0])
print(mns.train.labels[0])

X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

