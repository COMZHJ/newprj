import tensorflow as tf
from skimage import transform
import os
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm


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


# 定义样本路径
data_dir = 'E:/Source/picrecord'
# 载入文件名称与标签
(filenames, labels), _ = load_sample(data_dir, shuffleflag=False)
print(filenames, labels)


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






