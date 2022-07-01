#!/usr/bin/python3

# ============================ tabletop03_blackMat_shadow ==============================
"""
    @brief:         Test whether the black blackground will be similar to shadow, 
                    thereby make the shadow detection easier

                    The test will calculate the difference between the shadow color
                    and the black background color and white background color separately,
                    the one with smaller difference wins

    @author:    Yiye        yychen2019@gatech.edu
    @date:      09/11/2021
"""
# ============================ tabletop03_blackMat_shadow ==============================

# ====== [0] setup the environment. Read the data
from Surveillance.layers.base import Params
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

import detector.bgmodel.bgmodelGMM as BG
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.human_seg import Params as hParams

# ======= [0]  setup
fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/black_mat_shadow')

pure_white_video = cv2.VideoCapture(os.path.join(dPath, 'pure_whiteBG.avi'))
pure_black_video = cv2.VideoCapture(os.path.join(dPath, 'pure_blackBG.avi'))
shadow_white_files = []
shadow_black_files = []
for i in range(4):
    shadow_white_files.append(
        os.path.join(dPath, "handWave_whiteBG_{}.png".format(i))
    )
    shadow_black_files.append(
        os.path.join(dPath, "handWave_blackBG_{}.png".format(i))
    )

# bg extractor
bg_params_white = BG.Params_cv(
    history=300,
    NMixtures=5,
    varThreshold=30.,
    detectShadows=True,
    ShadowThreshold=0.5,
)
bg_params_black = deepcopy(bg_params_white)
bg_params_black.ShadowThreshold= 0.0
bg_extractor_white = BG.bgmodelGMM_cv(params=bg_params_white)
bg_extractor_black = BG.bgmodelGMM_cv(params=bg_params_black)

# train the params
bg_extractor_white.doAdapt = True
bg_extractor_black.doAdapt = True

ret = True
while(pure_white_video.isOpened() and ret):
    ret, frame = pure_white_video.read()
    if ret:
        bg_extractor_white.process(frame)
ret = True
while(pure_black_video.isOpened() and ret):
    ret, frame = pure_black_video.read()
    if ret:
        bg_extractor_black.process(frame)

# try to detect the shadow
bg_extractor_black.doAdapt = False
bg_extractor_white.doAdapt = False
#for i in range(4):
for i in [1]:
    shadow_white_img = cv2.imread(shadow_white_files[i])[:,:,::-1]
    bg_extractor_white.process(shadow_white_img)
    fh, axes = plt.subplots(1, 3, figsize=(15, 5))
    fh.suptitle("The varThreshold={}. ShadowThreshold={}".\
        format(bg_params_white.varThreshold, bg_params_white.ShadowThreshold))
    axes[0].imshow(shadow_white_img)
    axes[1].imshow(bg_extractor_white.shadow_mask[:,:,np.newaxis] * shadow_white_img)
    axes[1].set_title("Shadow region")
    axes[2].imshow(bg_extractor_white.fg_mask[:,:,np.newaxis] * shadow_white_img)
    axes[2].set_title("Foreground region")

    shadow_black_img = cv2.imread(shadow_black_files[i])[:,:,::-1]
    bg_extractor_black.process(shadow_black_img)
    fh, axes = plt.subplots(1, 3, figsize=(15, 5))
    fh.suptitle("The varThreshold={}. ShadowThreshold={}".\
        format(bg_params_black.varThreshold, bg_params_black.ShadowThreshold))
    axes[0].imshow(shadow_black_img)
    axes[1].imshow(bg_extractor_black.shadow_mask[:,:,np.newaxis] * shadow_black_img)
    axes[1].set_title("Shadow region")
    axes[2].imshow(bg_extractor_black.fg_mask[:,:,np.newaxis] * shadow_black_img)
    axes[2].set_title("Foreground region")


plt.show()


