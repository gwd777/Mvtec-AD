import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def calculate_histogram(image_path, is_color=True):
    # 读取图像
    image = Image.open(image_path)
    if not is_color:
        image = image.convert('L')

    # 计算直方图
    if is_color:
        r, g, b = image.split()
        hist_r = np.array(r.histogram())
        hist_g = np.array(g.histogram())
        hist_b = np.array(b.histogram())
        return hist_r, hist_g, hist_b
    else:
        hist = np.array(image.histogram())
        return hist


def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    # 将PIL直方图转换为OpenCV格式
    hist1 = np.float32(hist1)
    hist2 = np.float32(hist2)

    # 比较直方图
    return cv2.compareHist(hist1, hist2, method)


# 计算两个图像的直方图
hist1_gray = calculate_histogram('9花纱张力偏紧-86_0492_pic_1681906271245_camera01.png', is_color=False)
hist2_gray = calculate_histogram('14氨纶双丝-86_0024_pic_1681719258570_camera14.png', is_color=False)

# 比较灰度直方图的相似性
similarity = compare_histograms(hist1_gray, hist2_gray, method=cv2.HISTCMP_CORREL)
print(f'Grayscale Histogram Similarity: {similarity}')

# 如果是彩色图像
hist1_color = calculate_histogram('000.png', is_color=True)
hist2_color = calculate_histogram('001.png', is_color=True)

# 比较每个通道的直方图相似性
similarity_r = compare_histograms(hist1_color[0], hist2_color[0], method=cv2.HISTCMP_CORREL)
similarity_g = compare_histograms(hist1_color[1], hist2_color[1], method=cv2.HISTCMP_CORREL)
similarity_b = compare_histograms(hist1_color[2], hist2_color[2], method=cv2.HISTCMP_CORREL)

print(f'Red Channel Similarity: {similarity_r}')
print(f'Green Channel Similarity: {similarity_g}')
print(f'Blue Channel Similarity: {similarity_b}')
