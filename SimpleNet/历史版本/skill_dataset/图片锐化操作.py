import cv2
from PIL import Image, ImageFilter
import numpy as np

# 读取图像
image = Image.open('14氨纶双丝-86_0024_pic_1681719258570_camera14.png')

# 检查图像模式，如果是调色板模式(P mode)或L模式，转换为RGB模式
if image.mode in ('P', 'L'):
    image = image.convert('RGB')

# 定义自定义锐化滤波器
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  10, -1],
                              [-1, -1, -1]])

# 应用自定义滤波器
sharpened_image = image.filter(ImageFilter.Kernel((3, 3), sharpening_kernel.flatten(), 1, 0))

# 显示结果
sharpened_image.show()

# 保存锐化后的图像
sharpened_image.save('custom_sharpened_image.jpg')
