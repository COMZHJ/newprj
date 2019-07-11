import tensorflow as tf
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

record_path = '../data/mydata.tfrecords'


# 定义生成TFRecord的函数
def makeTFRec(filenames, labels):
    # 定义writer，用于向TFRecords文件写入数据
    writer = tf.python_io.TFRecordWriter(record_path)
    for i in tqdm(range(0, len(labels))):
        img = Image.open(filenames[i])
        img = img.resize((256, 256))
        # 将图片转化为二进制格式
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
        # 序列化为字符串
        writer.write(example.SerializeToString())

    # 数据集制作完成
    writer.close()


# 生成TFRecord数据集
makeTFRec(filenames, labels)


# 将TFRecord数据集转化为可以输入静态图的队列格式
def read_and_decode(filenames, flag='train', batch_size=3):
    # 根据文件名生成一个队列
    if flag == 'train':
        # 乱序操作，并循环读取
        filename_queue = tf.train.string_input_producer(filenames)
    else:
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)

    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    # 取出包含image和label的feature
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64), 'img_raw': tf.FixedLenFeature([], tf.string)})

    # tf.decode_raw 可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [256, 256, 3])
    # 转换标签类型
    label = tf.cast(features['label'], tf.int32)

    # 如果是训练使用，则应将其归一化，并按批次组合
    if flag == 'train':
        # 归一化
        image = tf.cast(image, tf.float32) * (1./255) - 0.5
        # 按照批次组合
        img_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=20)
        return img_batch, label_batch

    return image, label


TFRecordfilenames = [record_path]
# 以测试的方式打开数据集
image, label = read_and_decode(TFRecordfilenames, flag='test')
print(image, label)

# 定义保存图片的路径
saveimgpath = 'E:/Source/pic/show/'
# 如果存在saveimgpath，则将其删除
if tf.gfile.Exists(saveimgpath):
    tf.gfile.DeleteRecursively(saveimgpath)
# 创建saveimgpath路径
tf.gfile.MakeDirs(saveimgpath)

# 开始一个读取数据的会话
with tf.Session() as sess:
    # 初始化本地变量，没有这句会报错
    init = tf.local_variables_initializer()
    sess.run(init)

    # 建立列队协调器（启动多线程）
    coord = tf.train.Coordinator()
    # 启动队列线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 建立集合对象，用于存放子文件夹
    myset = set([])

    try:
        i = 0
        while True:
            # 取出image和label
            example, examplelab = sess.run([image, label])
            examplelab = str(examplelab)
            # 创建文件夹
            if examplelab not in myset:
                myset.add(examplelab)
                tf.gfile.MakeDirs(saveimgpath + examplelab)

            # 转换Image格式
            img = Image.fromarray(example, 'RGB')
            # 保存图片
            img.save(saveimgpath + examplelab + '/' + str(i) + '_Label_' + '.jpg')

            print(i)
            i = i + 1

    except tf.errors.OutOfRangeError:
        print('Done Test -- epoch limit reached')
    finally:
        coord.request_stop()
        # 关闭队列
        coord.join(threads)
        print('stop()')

