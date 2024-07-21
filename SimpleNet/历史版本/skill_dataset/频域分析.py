import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread('14氨纶双丝-86_0024_pic_1681719258570_camera14.png', 0)
image = np.array(image)

# 进行傅里叶变换
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 构建增益矩阵
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2
mask = np.zeros((rows, cols, 2), np.float32)
r = 30
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = np.sqrt((x - center[0])**2 + (y - center[1])**2) <= r
mask[mask_area] = 2  # 增强频率成分

# 应用增益矩阵
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('原始图像'), plt.axis('off')
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('增强频率后的图像'), plt.axis('off')
plt.show()
