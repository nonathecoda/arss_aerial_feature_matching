from icecream import ic
from kornia.feature import LoFTR
from PIL import Image
import numpy as np
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
import cv2
import tensorflow as tf
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

cam1 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam1/cam1_00138_e_0010_g_01_87968505_corr_rect.tiff"
cam2 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam2/cam2_00138_e_0010_g_01_87968505_corr_rect.tiff"
cam3 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam3/cam3_00138_e_0010_g_01_87968505_corr_rect.tiff"
cam4 = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/cam4/cam4_00138_e_0010_g_01_87968505_corr_rect.tiff"
camT = "/Users/antonia/dev/UNITN/remote_sensing_systems/data/ARSS_P3/camT/camT_00127_00084436_rect.tiff"

#cam 1-4 / 2-3 / 2-4 / 3-4 very goood

fname1 = cam2
fname2 = camT

img1 = K.io.load_image(fname1, K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image(fname2, K.io.ImageLoadType.RGB32)[None, ...]

img1 = K.geometry.resize(img1, (240, 320), antialias=True)
img2 = K.geometry.resize(img2, (240, 320), antialias=True)

matcher = KF.LoFTR(pretrained="outdoor")

input_dict = {
    "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
    "image1": K.color.rgb_to_grayscale(img2),
}

with torch.inference_mode():
    correspondences = matcher(input_dict)

mkpts0 = correspondences["keypoints0"].cpu().numpy()
mkpts1 = correspondences["keypoints1"].cpu().numpy()
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
    ax = ax
)
plt.show()
plt.savefig('result.png')

