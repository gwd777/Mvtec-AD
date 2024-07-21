import cv2
import numpy as np
from PIL import Image

# 读取图像
# image = cv2.imread('input_image.jpg')

image = Image.open('018.png')
image = np.array(image)

# 定义高通滤波器
kernel_size = 3
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])

# 对图像应用高通滤波器
sharpened_image = cv2.filter2D(image, -1, kernel)

# 显示原始图像和锐化后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
