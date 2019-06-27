import numpy as np
from PIL import Image

im = Image.open('../pic/phone.jpg')

print(im, type(im))
im = np.array(im)

print(im, type(im), im.shape, im.dtype)
im = [255, 255, 255] - im
print(im, type(im), im.shape, im.dtype)
im = Image.fromarray(im.astype(np.uint8))

im.save('../pic/phone2.jpg')

