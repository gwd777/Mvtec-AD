import json
import numpy as np
from PIL import Image, ImageDraw


def generate_mask(json_file):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建一个全0的掩码图像
    width = data["imageWidth"]
    height = data["imageHeight"]
    mask = np.zeros((height, width), dtype=np.uint8)

    # 遍历每个形状,在掩码上绘制
    for shape in data["shapes"]:
        points = [(int(x), int(y)) for x, y in shape["points"]]  # 确保坐标点为整数
        poly = Image.new('L', (width, height), 0)
        pdraw = ImageDraw.Draw(poly)
        pdraw.polygon(points, fill=1, outline=1)
        mask += np.array(poly, dtype=np.uint8)

    # 确保掩码值在0和1之间
    mask = np.clip(mask, 0, 1)
    mask_img = Image.fromarray(mask * 255).convert('L')
    return mask_img


# 使用示例
filename = r'F:\Data\HW\FZYB\leisi\anomaly\simple\2断贾卡纱-86_0032_pic_1681719730233_camera24.png'
mask_path = filename.replace('png', 'json')
mask = generate_mask(mask_path)

print(filename, mask_path, sep='\n')
mask