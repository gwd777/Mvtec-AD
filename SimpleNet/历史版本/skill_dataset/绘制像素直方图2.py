from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像并计算直方图
image_gray = Image.open('14氨纶双丝-86_0024_pic_1681719258570_camera14.png').convert('L')
hist_gray = image_gray.histogram()

# 绘制灰度直方图
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
plt.plot(hist_gray)
plt.xlim([0, 256])
plt.show()

# 读取彩色图像并计算直方图
image_color = Image.open('14氨纶双丝-86_0024_pic_1681719258570_camera14.png')
r, g, b = image_color.split()
hist_r = r.histogram()
hist_g = g.histogram()
hist_b = b.histogram()

# 绘制彩色直方图
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
plt.plot(hist_r, color='red', label='Red Channel')
plt.plot(hist_g, color='green', label='Green Channel')
plt.plot(hist_b, color='blue', label='Blue Channel')
plt.xlim([0, 256])
plt.legend()
plt.show()
