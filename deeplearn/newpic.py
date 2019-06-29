# 采用可视化再一张大图中显示多张小图
import matplotlib.pyplot as plt
from PIL import Image

# 设置尺寸和分别率
plt.figure(figsize=[30, 20], dpi=100)
# 指定布局和当前图片显示再布局的位置
im = Image.open('../data/phone.jpg')

for index in range(1, 56):
    plt.subplot(5, 11, index)
    plt.imshow(im)

plt.show()

