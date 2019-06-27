import numpy as np

t1 = np.arange(12).reshape(3, 4)
t1[0, 0] = 100
print(t1, t1.shape)
print(t1[0:2, 1:])
print(t1[0:2, [1, 3]])
print(t1[[0, 2], [1, 3]])

t2 = t1[:, [1, 3]]
print(t2, t2.shape)
