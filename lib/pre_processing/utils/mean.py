import os
import numpy as np
from PIL import Image

data_folder = '/home/cidi/桌面/semi_keypoint/data/09_01/train/imgs'
mean = [0, 0, 0]
std = [0, 0, 0]
num_images = 0

# 遍历数据集文件夹中的图像
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):  # 根据实际图像格式进行调整
            image_path = os.path.join(root, file)
            img = Image.open(image_path)
            img_array = np.array(img).astype(np.float32) / 255.0  # 将像素值缩放到 [0, 1]
            mean += img_array.mean(axis=(0, 1))
            std += img_array.std(axis=(0, 1))
            num_images += 1

mean /= num_images
std /= num_images

print("Mean:", mean)
print("Std:", std)