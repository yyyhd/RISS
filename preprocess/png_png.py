import os
import numpy as np
from PIL import Image

png_folder1 = ''
png_folder2 = ''
result = ''

# 获取两个文件夹中所有 PNG 文件的列表
png_files1 = [f for f in os.listdir(png_folder1) if f.endswith('.png')]
png_files2 = [f for f in os.listdir(png_folder2) if f.endswith('.png')]

# 确保两个文件夹中的文件数量相同
assert len(png_files1) == len(png_files2),
count = 0
# 循环处理每一对 PNG 文件
for i in range(len(png_files1)):
    patient_id = os.path.splitext(png_files1[i])[0]
    # 构建 PNG 文件的完整路径
    png_path1 = os.path.join(png_folder1, png_files1[i])
    png_path2 = os.path.join(png_folder2, png_files2[i])

    # 从 PNG 文件加载图像数据，并直接转换为灰度图像
    img1 = Image.open(png_path1).convert('L')
    img2 = Image.open(png_path2).convert('L')

    # 转换为 NumPy 数组
    img_arr1 = np.array(img1)
    img_arr2 = np.array(img2)

    # 将数据类型转为uint8，并调整像素值范围到0-255
    t1_img = (img_arr1 / np.max(img_arr1) * 255).astype(np.uint8)
    t2_img = (img_arr2 / np.max(img_arr2) * 255).astype(np.uint8)

    zeros_channel = np.zeros_like(t1_img)
    # 将三个模态的数据合并成一张图片，每个模态的数据存储在不同的通道中
    img1 = np.stack((t1_img, zeros_channel, zeros_channel), axis=-1)
    img2 = np.stack((zeros_channel, t2_img, zeros_channel), axis=-1)
    img = np.concatenate((img1, img2), axis=1)
    img = Image.fromarray(np.uint8(img))

    count += 1
    # 构建输出的.png文件名，以病人id命名
    # img_name = f'{patient_id}_1.png'#cc
    # img_name = f'{patient_id}_mlo.png'#cc
    img_name = f'{patient_id}.png'#mlo

    # 保存图片
    img.save(os.path.join(result, img_name))

    print("已处理" + str(count) + "个")