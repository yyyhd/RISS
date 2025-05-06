import os
import cv2
import numpy as np
from scipy import signal
from enum import Enum

real_folder = "Evaluation/real"
fake_folder = "Evaluation/syn"


def filter2(img, fltr, mode='same'):
    return signal.convolve2d(img, np.rot90(fltr, 2), mode=mode)


def _get_sums(GT, P, win, mode='same'):
    mu1, mu2 = (filter2(GT, win, mode), filter2(P, win, mode))
    return mu1 * mu1, mu2 * mu2, mu1 * mu2


def _get_sigmas(GT, P, win, mode='same', **kwargs):
    if 'sums' in kwargs:
        GT_sum_sq, P_sum_sq, GT_P_sum_mul = kwargs['sums']
    else:
        GT_sum_sq, P_sum_sq, GT_P_sum_mul = _get_sums(GT, P, win, mode)

    return filter2(GT * GT, win, mode) - GT_sum_sq, \
           filter2(P * P, win, mode) - P_sum_sq, \
           filter2(GT * P, win, mode) - GT_P_sum_mul


class Filter(Enum):
    UNIFORM = 0
    GAUSSIAN = 1


def fspecial(fltr, ws, **kwargs):
    if fltr == Filter.UNIFORM:
        return np.ones((ws, ws)) / ws ** 2
    elif fltr == Filter.GAUSSIAN:
        x, y = np.mgrid[-ws // 2 + 1: ws // 2 + 1, -ws // 2 + 1: ws // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * kwargs['sigma'] ** 2)))
        g[g < np.finfo(g.dtype).eps * g.max()] = 0
        assert g.shape == (ws, ws)
        den = g.sum()
        if den != 0:
            g /= den
        return g
    return None


def _ssim_single(GT, P, ws, C1, C2, fltr_specs, mode):
    win = fspecial(**fltr_specs)

    GT_sum_sq, P_sum_sq, GT_P_sum_mul = _get_sums(GT, P, win, mode)
    sigmaGT_sq, sigmaP_sq, sigmaGT_P = _get_sigmas(GT, P, win, mode, sums=(GT_sum_sq, P_sum_sq, GT_P_sum_mul))

    assert C1 > 0
    assert C2 > 0

    ssim_map = ((2 * GT_P_sum_mul + C1) * (2 * sigmaGT_P + C2)) / (
            (GT_sum_sq + P_sum_sq + C1) * (sigmaGT_sq + sigmaP_sq + C2))
    cs_map = (2 * sigmaGT_P + C2) / (sigmaGT_sq + sigmaP_sq + C2)

    return np.mean(ssim_map), np.mean(cs_map)


def ssim(GT, P, ws=11, K1=0.01, K2=0.03, MAX=None, fltr_specs=None, mode='valid'):
    if MAX is None:
        MAX = np.iinfo(GT.dtype).max

    assert GT.shape == P.shape, "Supplied images have different sizes " + \
                                str(GT.shape) + " and " + str(P.shape)

    if fltr_specs is None:
        fltr_specs = dict(fltr=Filter.UNIFORM, ws=ws)

    C1 = (K1 * MAX) ** 2
    C2 = (K2 * MAX) ** 2

    ssims = []
    css = []
    for i in range(GT.shape[2]):
        ssim, cs = _ssim_single(GT[:, :, i], P[:, :, i], ws, C1, C2, fltr_specs, mode)
        ssims.append(ssim)
        css.append(cs)
    return np.mean(ssims), np.mean(css)


def get_matching_filenames(folder1, folder2):
    files1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    files2 = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

    matching_files = []
    for file1 in files1:
        for file2 in files2:
            if file1[:16] == file2[:16]:  # Check if the first 11 characters match
                matching_files.append((os.path.join(folder1, file1), os.path.join(folder2, file2)))
                break

    return matching_files

def calculate_ssim_for_folder_pairs(real_folder, fake_folder):
    matching_files = get_matching_filenames(real_folder, fake_folder)

    all_img_sims = []

    for real_img, fake_img in matching_files:
        img1 = cv2.imread(real_img)
        img2 = cv2.imread(fake_img)

        img_sim, _ = ssim(img1, img2)
        all_img_sims.append(img_sim)
        print(f"SSIM for {os.path.basename(real_img)} and {os.path.basename(fake_img)}: {img_sim}")

    average_img_sim = np.mean(all_img_sims)
    std_dev_img_sim = np.std(all_img_sims)
    median_img_sim = np.median(all_img_sims)
    q75, q25 = np.percentile(all_img_sims, [75 ,25])
    iqr = q75 - q25
    
    print(f"Average SSIM for all image pairs: {average_img_sim}")
    print(f"Standard deviation of SSIM values: {std_dev_img_sim}")
    print(f"Median SSIM for all image pairs: {median_img_sim}")
    print(f"Interquartile range (IQR) of SSIM values: {iqr}")
    print(f"Q75: {q75}, Q25: {q25}")
calculate_ssim_for_folder_pairs(real_folder, fake_folder)


