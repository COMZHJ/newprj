import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


# 创建数据集
def dataset(directory, size, batchsize):
    # 解析一个图片文件
    def _parseone(example_photo):
        # 定义解析的字典
        dics = {}
        dics['label'] = tf.FixedLenFeature(shape=[], dtype=tf.int64)
        dics['img_raw'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
        # 解析一行样本
        parsed_example = tf.parse_single_example(example_photo, dics)

        image = tf.decode_raw(parsed_example['img_raw'], out_type=tf.uint8)
        image = tf.reshape(image, size)
        # 对图像数据做归一化处理
        image = tf.cast(image, tf.float32) * (1./255) - 0.5

        label = parsed_example['label']
        label = tf.cast(label, tf.int32)
        # 转为One-hot编码
        label = tf.one_hot(label, depth=2, on_value=1)
        return image, label

    # 生成Dataset对象
    dataset = tf.data.TFRecordDataset(directory)
    # 转化为图片数据集
    dataset = dataset.map(_parseone)
    # 批次组合数据集
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(batchsize)

    return dataset


# 显示单个图片
def showresult(subplot, title, thisimg):
    p = plt.subplot(subplot)
    p.axis('off')

    p.imshow(thisimg)
    p.set_title(title)


# 显示批次图片
def showimg(index, label, img, ntop):
    # 定义显示图片的宽和高
    plt.figure(figsize=(20, 10))
    plt.axis('off')
    ntop = min(ntop, 9)

    print(index)
    for i in range(ntop):
        showresult(100 + 10*ntop + 1 + i, label[i], img[i])
    plt.show()


def getone(dataset):
    # 生成一个迭代器
    iterator = dataset.make_one_shot_iterator()
    # 从iterator里取出一个元素
    one_element = iterator.get_next()
    return one_element


sample_dir = '../data/mydata.tfrecords'
size = [256, 256, 3]
batchsize = 10
tdataset = dataset(sample_dir, size, batchsize)
# 打印数据集的输出信息
print(tdataset.output_types)
print(tdataset.output_shapes)

# 取出一个元素
one_element = getone(tdataset)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  # 初始化

    try:
        for step in np.arange(1):
            value = sess.run(one_element)
            # 显示图片
            showimg(step, value[1], np.asarray((value[0]+0.5)*255, np.uint8), 10)
            print(step)

    # 捕获异常
    except tf.errors.OutOfRangeError:
        print('Done!!!')

