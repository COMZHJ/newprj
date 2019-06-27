import numpy as np

t1 = np.arange(12)
print(t1, type(t1), t1.shape)
t1 = t1.reshape(3, 4)
print(t1, type(t1), t1.shape)

print('矩阵常见的属性')
print('矩阵的形状', t1.shape, '元素数量', t1.size)
print(f'数据类型：{t1.dtype}，元素所占空间：{t1.itemsize}')
t1 = t1.astype(np.float64)
print(f'数据类型：{t1.dtype}，元素所占空间：{t1.itemsize}')

# 矩阵的运算，每个元素都单独计算
print(t1 + 1)

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(data, type(data), len(data))
t1 = np.array(data)
print(t1, type(t1), t1.shape)

# import random
t1 = np.random.random(size=[3, 4])
print(t1, type(t1), t1.shape)
t1 = np.random.randint(1, 20, [3, 4])
print(t1, type(t1), t1.shape)
