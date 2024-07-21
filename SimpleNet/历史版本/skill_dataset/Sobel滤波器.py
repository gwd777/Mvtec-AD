from PIL import Image, ImageFilter, ImageOps
import numpy as np

# 读取图像并转换为灰度图像
image = Image.open('custom_sharpened_image.jpg').convert('L')

# 应用高斯滤波
blurred = image.filter(ImageFilter.GaussianBlur(2))

# 将图像转换为NumPy数组
blurred_np = np.array(blurred)

# 使用Sobel算子进行边缘检测
sobelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

grad_x = np.abs(np.convolve(blurred_np.flatten(), sobelx.flatten(), 'same').reshape(blurred_np.shape))
grad_y = np.abs(np.convolve(blurred_np.flatten(), sobely.flatten(), 'same').reshape(blurred_np.shape))

edges = np.sqrt(grad_x**2 + grad_y**2)
edges = (edges / np.max(edges) * 255).astype(np.uint8)

# 应用二值化
binary_edges = (edges > 50) * 255
binary_edges = binary_edges.astype(np.uint8)

# 将边缘图像转换回PIL图像
binary_edges_img = Image.fromarray(binary_edges)

# 将二值化后的边缘叠加到原图像
highlighted_texture = Image.blend(image, binary_edges_img, alpha=0.5)

# 显示结果
highlighted_texture.show()
