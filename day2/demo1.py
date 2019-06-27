import numpy as np

t1 = np.arange(12).reshape(3, 4)
t1[0, 0] = 100
print(t1, t1.shape)
print(t1[0:2, 1:])
print(t1[0:2, [1, 3]])


