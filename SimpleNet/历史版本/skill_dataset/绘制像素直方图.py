from PIL import Image
import matplotlib.pyplot as plt

# 读取图像
image = Image.open('14氨纶双丝-86_0024_pic_1681719258570_camera14.png')

# 如果图像是彩色的，将其转换为灰度图像
if image.mode != 'L':
    image = image.convert('L')

# 获取像素值
pixels = list(image.getdata())

# 绘制直方图
plt.hist(pixels, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Pixel Values')
plt.show()
