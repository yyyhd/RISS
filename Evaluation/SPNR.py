import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 指定两个文件夹路径分别存储PNG格式图像
folder1 = "Evaluation/real"
folder2 = "Evaluation/syn"
# 获取文件夹中所有文件名
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

ssim_values = []
psnr_values = []

# 初始化最大和最小PSNR值  
max_psnr = float('-inf')  
min_psnr = float('inf')  

# 计算两个文件夹中所有图像的PSNR
for file1 in files1:
    if file1.endswith(".png"):
        file_prefix = file1[:11]  # 获取文件名中的前十六位作为前缀
        corresponding_file = [file2 for file2 in files2 if file2.startswith(file_prefix)]

        if len(corresponding_file) > 0:
            img1 = cv2.imread(os.path.join(folder1, file1))
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            img2 = cv2.imread(os.path.join(folder2, corresponding_file[0]))  # 使用匹配的文件计算和PSNR
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # 计算均方误差
            mse = np.mean((gray1 - gray2) ** 2)
            # 计算峰值信噪比
            psnr = 20 * np.log10(255 / np.sqrt(mse))
            psnr_values.append(psnr)

            # 更新最大和最小PSNR值
            if psnr > max_psnr:
                max_psnr = psnr
            if psnr < min_psnr:
                min_psnr = psnr

            print(f"PSNR between {file1} and {corresponding_file[0]}: {psnr}")

# 计算PSNR值的平均值、标准差、中位数及四分位数范围
average_psnr = np.mean(psnr_values)
std_dev_psnr = np.std(psnr_values)
median_psnr = np.median(psnr_values)
q75, q25 = np.percentile(psnr_values, [75, 25])
iqr = q75 - q25

# 打印统计结果
print(f"Maximum PSNR: {max_psnr}")
print(f"Minimum PSNR: {min_psnr}")
print(f"Average PSNR: {average_psnr}")
print(f"Standard deviation of PSNR values: {std_dev_psnr}")
print(f"Median PSNR: {median_psnr}")
print(f"Interquartile range (IQR) of PSNR values: {iqr}")
print(f"Q75: {q75}, Q25: {q25}")
