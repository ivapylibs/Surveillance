#!/usr/bin/python3
# ============================ tabletop01 ==============================
"""
    @brief:         Use the GMM-based background substraction model to extract
                    the tabletop layer.
                    During the calibration phase, attempts to use the human segmenter
                    to get the ground truth foreground mask, so the background mask
                    will include both the tabletop and the shadow casted by the human 

    @author:    Yiye        yychen2019@gatech.edu
    @date:      08/26/2021
"""
# ============================ tabletop01 ==============================

# ====== [1] setup the environment. Read the data
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import detector.bgmodel.bgmodelGMM as BG
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange

fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/BG')

# data - tabletop train and test
bg_pure = cv2.VideoCapture(os.path.join(dPath, 'bgTrain_human_wave.avi'))
bg_test_files = []
for i in range(5):
    test_file = os.path.join(dPath, "bgTest_human_puzzle_{}.png".format(i))
    bg_test_files.append(test_file)

# data - human segmenter
train_img_glove = cv2.imread(os.path.join(dPath, "calibrate_glove_0.png"))[:,:,::-1]
train_depth_table = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"), allow_pickle=True
)["depth_frame"]
intrinsic = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"), allow_pickle=True
)["intrinsics"]


# ==== [2] Prepare the bg modeler & human segmenter
bg_params = BG.Params_cv(
    history=300,
    NMixtures=5,
    varThreshold=50.,
    detectShadows=True,
    ShadowThreshold=0.55,
)
bg_extractor = BG.bgmodelGMM_cv(params=bg_params)

# human segmenter
human_seg = Human_ColorSG_HeightInRange.buildFromImage(train_img_glove, train_depth_table, \
    intrinsic)

# ==== [3] Learn the GMM parameters
bg_extractor.doAdapt = True
ret=True
while(bg_pure.isOpened() and ret):
    ret, frame = bg_pure.read()
    if ret:
        # Get GT BG mask
        BG_mask = None

        # process with the GT BG mask
        bg_extractor.process(frame, BG_mask)

print(bg_extractor.get("NMixtures"))

# ==== [4] Test on the test data
bg_extractor.doAdapt = False
ret=True
for test_file in bg_test_files:
    test_img = cv2.imread(test_file)[:,:,::-1]
    bg_extractor.process(test_img)
    fgMask = bg_extractor.getForeground()
    detResult = bg_extractor.getDetectResult() 
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(test_img)
    axes[0].set_title("The test image")
    axes[1].imshow(detResult, cmap='gray') 
    axes[1].set_title("The detected Foreground(white) and Shadow(gray)")
    axes[2].imshow(fgMask, cmap='gray')
    axes[2].set_title("The foreground")

    print(bg_extractor.get("NMixtures"))
plt.show()