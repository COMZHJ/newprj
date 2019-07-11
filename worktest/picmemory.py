import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def load_sample(sample_dir):
    # 递归读取文件，只支持一级，返回文件名、数值标签、数值对应的标签名
    print('loading sample dataset...')
    lfilenames = []
    labelsnames = []

    # 遍历文件夹
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):
        for filename in filenames:  # 遍历所有文件名
            filename_path = os.sep.join([dirpath, filename])
            # 添加文件名
            lfilenames.append(filename_path)
            # 添加文件名对应的标签
            labelsnames.append(dirpath.split('\\')[-1])

    # 生成标签名称列表
    lab = list(sorted(set(labelsnames)))
    # 生成字典
    labdict = dict(zip(lab, list(range(len(lab)))))

    labels = [labdict[i] for i in labelsnames]
    return shuffle(np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


data_dir = 'E:/Source/pic'

(image, label), labelsnames = load_sample(data_dir)
print(len(image), image[:2], len(label), label[:2])
print(labelsnames[label[:2]], labelsnames)


def get_batches(image, label, resize_w, resize_h, channels, batch_size):
    # 实现一个输入队列
    queue = tf.train.slice_input_producer([image, label])
    # 从输入队列里读取标签
    label = queue[1]
    # 从输入队列里读取image路径
    image_c = tf.read_file(queue[0])
    # 按照路径读取图片
    image = tf.image.decode_bmp(image_c, channels)

    # 修改图片大小
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
    # 将图像进行标准化处理
    image = tf.image.per_image_standardization(image)
    # 生成批次数据
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64)

    # 将数据类型转换为float32
    images_batch = tf.cast(image_batch, tf.float32)
    # 修改标签的形状
    labels_batch = tf.reshape(label_batch, [batch_size])
    return images_batch, labels_batch


batch_size = 16
image_batches, label_batches = get_batches(image, label, 50, 30, 1, batch_size)


# 显示单个图片
def showresult(subplot, title, thisimg):
    p = plt.subplot(subplot)
    p.axis('off')

    p.imshow(np.reshape(thisimg, (50, 30)))
    p.set_title(title)


# 显示批次图片
def showimg(index, label, img, ntop):
    # 定义显示图片的宽和高
    plt.figure(figsize=(20, 10))
    p.axis('off')
    ntop = min(ntop, 9)

    print(index)
    for i in range(ntop):
        showresult(100 + 10*ntop + 1 + i, label[i], img[i])
    plt.show()


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  # 初始化

    # 建立列队协调器
    coord = tf.train.Coordinator()
    # 启动队列线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(10):
            if coord.should_stop():
                break
            # 注入数据
            images, label = sess.run([image_batches, label_batches])
            # 显示图片
            showimg(step, label, images, batch_size)
            print(label)

    except tf.errors.OutOfRangeError:
        print('Done!!!')
    finally:
        coord.request_stop()

    # # 关闭队列
    # coord.join(threads)

