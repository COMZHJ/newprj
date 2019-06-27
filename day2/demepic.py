import numpy as np
from PIL import Image

im = Image.open('../data/phone.jpg')

print(im, type(im))
im = np.array(im)

print('原始矩阵', im, type(im), im.shape, im.dtype)
im = [255, 255, 255] - im
print('新矩阵', im, type(im), im.shape, im.dtype)
im = Image.fromarray(im.astype(np.uint8))

im.save('../pic/phone2.jpg')

