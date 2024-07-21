import cv2
import numpy as np
from PIL import Image

# 读取图像
# image = cv2.imread('14氨纶双丝-86_0024_pic_1681719258570_camera14.png', cv2.IMREAD_GRAYSCALE)
image = Image.open('custom_sharpened_image.jpg')

image = np.array(image)
# 应用高斯滤波
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 使用Sobel算子进行边缘检测
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

# 计算边缘幅度
edges = cv2.magnitude(sobelx, sobely)

# 转换为8位图像
edges = np.uint8(edges)

# 应用二值化
_, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

# 将二值化后的边缘叠加到原图像
highlighted_texture = cv2.addWeighted(image, 0.5, binary_edges, 0.5, 0)

# 显示结果
cv2.imshow('Highlighted Texture', highlighted_texture)
cv2.waitKey(0)
cv2.destroyAllWindows()
