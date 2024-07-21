import torch
import numpy as np


def calculate_histogram(image, bins=256):
    # 确保Tensor是浮点型
    image = image.float()

    # 计算直方图
    hist = torch.histc(image, bins=bins)

    # 将直方图转换为numpy数组
    hist = hist.numpy()

    return hist


if __name__ == '__main__':
    # 创建一个随机Tensor
    image_tensor = torch.rand(356, 256, 256)

    # 计算直方图
    hist = calculate_histogram(image_tensor, bins=256)
    print(hist)



