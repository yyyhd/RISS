import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

folder1 = "Evaluation/real"
folder2 = "Evaluation/syn"

files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

ssim_values = []
psnr_values = []

max_psnr = float('-inf')  
min_psnr = float('inf')  

for file1 in files1:
    if file1.endswith(".png"):
        file_prefix = file1[:11]  
        corresponding_file = [file2 for file2 in files2 if file2.startswith(file_prefix)]

        if len(corresponding_file) > 0:
            img1 = cv2.imread(os.path.join(folder1, file1))
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            img2 = cv2.imread(os.path.join(folder2, corresponding_file[0]))  
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            
            mse = np.mean((gray1 - gray2) ** 2)
            
            psnr = 20 * np.log10(255 / np.sqrt(mse))
            psnr_values.append(psnr)

            
            if psnr > max_psnr:
                max_psnr = psnr
            if psnr < min_psnr:
                min_psnr = psnr

            print(f"PSNR between {file1} and {corresponding_file[0]}: {psnr}")

average_psnr = np.mean(psnr_values)
std_dev_psnr = np.std(psnr_values)
median_psnr = np.median(psnr_values)
q75, q25 = np.percentile(psnr_values, [75, 25])
iqr = q75 - q25


print(f"Median PSNR: {median_psnr}")
print(f"Interquartile range (IQR) of PSNR values: {iqr}")
print(f"Q75: {q75}, Q25: {q25}")
