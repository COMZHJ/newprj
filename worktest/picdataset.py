import tensorflow as tf
from skimage import transform
import os
import numpy as np
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


def load_sample(sample_dir, shuffleflag=True):
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
    if shuffleflag == True:
        return shuffle(np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)
    else:
        return (np.asarray(lfilenames), np.asarray(labels)), np.asarray(lab)


# 随机变换图像
def _distorted_image(image, size, ch=1, shufferflag=False, cropflag=False,
                     brightnessflag=False, contrastflag=False):
    distorted_image = tf.image.random_flip_left_right(image)

    if cropflag == True:
        # 随机裁剪
        s = tf.random_uniform((1, 2), int(size[0] * 0.8), size[0], tf.int32)
        distorted_image = tf.random_crop(distorted_image, [s[0][0], s[0][0], ch])
    # 上下随机翻转
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    if brightnessflag == True:
        # 随机变化亮度
        distorted_image = tf.image.random_brightness(distorted_image, max_delta=10)
    if contrastflag == True:
        # 随机变化对比度
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    if shufferflag == True:
        # 沿着第0维打乱顺序
        distorted_image = tf.random_shuffle(distorted_image)

    return distorted_image


# 实现归一化，并且拍平
def _norm_image(image, size, ch=1, flattenflag=False):
    image_decoded = image/255.0
    if flattenflag == True:
        image_decoded = tf.reshape(image_decoded, [size[0] * size[1] * ch])
    return image_decoded


# 实现图片随机旋转操作
def _random_rotated30(image, label):
    # 封装好的skimage模块，将进行图片旋转30度
    def _rotated(image):
        shift_y, shift_x = np.array(image.shape.as_list()[:2], np.float32) / 2.
        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(30))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])

        image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
        return  image_rotated

    def _rotatedwrap():
        image_rotated = tf.py_function(_rotated, [image], [tf.float64])
        return tf.cast(image_rotated, tf.float32)[0]

    # 实现随机功能
    a = tf.random_uniform([1], 0, 2, tf.int32)
    image_decoded = tf.cond(tf.equal(tf.constant(0), a[0]), lambda: image, _rotatedwrap)

    return image_decoded, label


# 创建数据集
def dataset(directory, size, batchsize, random_rotated=False):
    # 载入文件名称与标签
    (filenames, labels), _ = load_sample(directory, shuffleflag=False)
    # 解析一个图片文件
    def _parseone(filename, label):
        # 读取并处理每张图片
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        # 对图片做扭曲变化
        image_decoded.set_shape([None, None, None])
        image_decoded = _distorted_image(image_decoded, size)
        # 变化尺寸
        image_decoded = tf.image.resize(image_decoded, size)
        # 归一化
        image_decoded = _norm_image(image_decoded, size)
        image_decoded = tf.cast(image_decoded, dtype=tf.float32)
        # 将label转为张量
        label = tf.cast(tf.reshape(label, []), dtype=tf.int32)
        return image_decoded, label

    # 生成Dataset对象
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # 转化为图片数据集
    dataset = dataset.map(_parseone)

    if random_rotated == True:
        dataset = dataset.map(_random_rotated30)
    # 批次组合数据集
    dataset = dataset.batch(batchsize)

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


sample_dir = 'E:/Source/picrecord'
size = [96, 96]
batchsize = 10
tdataset = dataset(sample_dir, size, batchsize)
tdataset2 = dataset(sample_dir, size, batchsize, True)
# 打印数据集的输出信息
print(tdataset.output_types)
print(tdataset.output_shapes)

# 取出一个元素
one_element = getone(tdataset)
one_element2 = getone(tdataset2)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  # 初始化

    try:
        for step in np.arange(1):
            value = sess.run(one_element)
            value2 = sess.run(one_element2)
            # 显示图片
            showimg(step, value[1], np.asarray(value[0]*255, np.uint8), 10)
            showimg(step, value2[1], np.asarray(value2[0]*255, np.uint8), 10)
            print(step)

    # 捕获异常
    except tf.errors.OutOfRangeError:
        print('Done!!!')

