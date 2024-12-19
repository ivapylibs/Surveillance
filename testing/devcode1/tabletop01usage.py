#!/usr/bin/python3
#============================ tabletop01usage ==============================
## @file
# @brief    Use the GMM-based background substraction model to extract the
#           tabletop layer.
#
# During the calibration phase, attempts to use the human segmenter to get the
# ground truth foreground mask, so the background mask will include both the
# tabletop and the shadow casted by the human 
# 
# @ingroup  TestSurveillance_Dev_v1
#
# @author   Yiye        yychen2019@gatech.edu
# @date     2021/08/26
#
# @quitf
#
#============================ tabletop01usage ==============================

# ====== [0] setup the environment. Read the data
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import detector.bgmodel.bgmodelGMM as BG
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.human_seg import Params as hParams
import Surveillance.layers.tabletop_seg as tabletop_seg

fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/BG')

# ======= [1] data
# bg train video
bg_hand = cv2.VideoCapture(os.path.join(dPath, 'bgTrain_human_wave.avi'))
# bg train depth. Here the result is pre-saved for future running
bg_hand_fgMask_path = os.path.join(dPath, "bgTrain_human_wave_fgMask.npz")
if os.path.exists(bg_hand_fgMask_path):
    bg_hand_depths = None
    bg_hand_fgMask = np.load(bg_hand_fgMask_path, allow_pickle="True")["fgMask"]
else:
    # if not pre-saved, then load the depths and run fg detection
    bg_hand_depths = np.load(
        os.path.join(dPath, "bgTrain_human_wave.npz"),
        allow_pickle=True
    )["depth_frames"]
    bg_hand_fgMask = np.zeros_like(bg_hand_depths, dtype=bool) 

bg_test_files = []
for i in range(5):
    test_file = os.path.join(dPath, "bgTest_human_puzzle_{}.png".format(i))
    bg_test_files.append(test_file)

# data - human segmenter
train_img_glove = cv2.imread(os.path.join(dPath, "calibrate_glove_0.png"))[:,:,::-1]
train_depth_table = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"), allow_pickle=True
)["depth_frame"]
train_img_table = cv2.imread(os.path.join(dPath, "empty_table_0.png"))[:,:,::-1]
intrinsic = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"), allow_pickle=True
)["intrinsics"]


# ==== [2] Prepare the bg modeler & human segmenter
bg_model_params = BG.Params_cv(
    history=300,
    NMixtures=5,
    varThreshold=20.,
    detectShadows=True,
    ShadowThreshold=0.55,
)
bg_seg_params = tabletop_seg.Params_GMM()
bg_extractor = tabletop_seg.tabletop_GMM.build(bg_model_params, bg_seg_params)

# human segmenter
params = hParams(det_th=20)
human_seg = Human_ColorSG_HeightInRange.buildFromImage(train_img_glove, train_depth_table, \
    intrinsic, params=params)

# ==== [3] Learn the GMM parameters
ret=True
idx = 0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("The training frames and the \"Ground Truth\" foreground mask")

display_interval = 20
print("The visualization of large images is slow. Slow only display one example every {} frames".format(display_interval))
while(bg_hand.isOpened() and ret):
    ret, frame = bg_hand.read()
    if ret:
        frame = frame[:,:,::-1]
        # Get GT BG & FG mask
        if bg_hand_depths is not None:
            depth = bg_hand_depths[idx,:,:]
            human_seg.update_depth(depth)
            human_seg.process(frame)
            fgMask = human_seg.get_mask()
            bg_hand_fgMask[idx, :, :] = fgMask
        else:
            fgMask = bg_hand_fgMask[idx, :, :]

        idx += 1
        BG_mask = ~fgMask

        # process with the GT BG mask
        frame_train = np.where(
            np.repeat(BG_mask[:,:,np.newaxis], 3, axis=2),
            frame, 
            train_img_table
        )
        bg_extractor.calibrate(frame_train)

        # visualize
        if idx % display_interval == 0:
            axes[0].imshow(frame)
            axes[0].set_title("The training frame")
            axes[1].imshow(fgMask, cmap="gray")
            axes[1].set_title("The GT foreground mask")
            axes[2].imshow(frame_train)
            axes[2].set_title("The synthetic training frame")
            plt.draw()
            plt.pause(0.01)
# save the fgmask result
if bg_hand_depths is not None:
    np.savez(bg_hand_fgMask_path, fgMask=bg_hand_fgMask)
    print("The FG mask saved out")

# ==== [4] Test on the test data
ret=True
for idx, test_file in enumerate(bg_test_files):
    test_img = cv2.imread(test_file)[:,:,::-1]

    # with shadow detection
    bg_extractor.process(test_img)
    fgMask = ~bg_extractor.get_mask()
    detResult = bg_extractor.detector.getDetectResult()     #<- THis will contain the shadow detection result as gray
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("The GMM with Shadow detection. Test frame: {}".format(idx))
    axes[0].imshow(test_img)
    axes[0].set_title("The test image")
    axes[1].imshow(detResult, cmap='gray') 
    axes[1].set_title("The detected Foreground(white) and Shadow(gray)")
    axes[2].imshow(fgMask, cmap='gray')
    axes[2].set_title("The foreground")

plt.show()

#
#============================ tabletop01usage ==============================
