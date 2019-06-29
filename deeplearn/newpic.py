# 采用可视化再一张大图中显示多张小图
import matplotlib.pyplot as plt
from PIL import Image

# 设置尺寸和分别率
plt.figure(figsize=[50, 30], dpi=10)
# 指定布局和当前图片显示再布局的位置
plt.subplot(5, 11, 1)
plt.imshow(Image.open('../data/phone.jpg'))
plt.show()


