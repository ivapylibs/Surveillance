#!/usr/bin/python3
#============================= puzzle01aruco =============================
## @file
# @brief    Puzzle layer top-down rectification based on the aruco tag 
# 
# The bird-eye-view transformation matrix is obtained from the aruco-tag.
# (pre-saved from the camera repository) The puzzle pieces mask is the remain
# of the background mask and the human mask
# 
# @ingroup  TestSurveillance_Dev_v1
#
# @version  v1.0 of Puzzlebot
# @author   Yiye        yychen2019@gatech.edu
# @date     2021/09/05
# 
# @quitf
#
#============================= puzzle01aruco =============================

# ====== [0] setup the environment. Read the data
from Surveillance.layers.base import Params
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label

import detector.bgmodel.bgmodelGMM as BG
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.human_seg import Params as hParams
import Surveillance.layers.puzzle_seg as Puzzle_Seg
from improcessor.mask import mask as maskproc
import trackpointer.centroidMulti as mCentroid


fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/puzzle_layer')

# ======= [1] data

# bg train video - for the background mask
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


# data - for the human segmenter
train_img_glove = cv2.imread(os.path.join(dPath, "calibrate_glove_0.png"))[:,:,::-1]
train_depth_table = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"), allow_pickle=True
)["depth_frame"]
train_img_table = cv2.imread(os.path.join(dPath, "empty_table_0.png"))[:,:,::-1]
intrinsic = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"), allow_pickle=True
)["intrinsics"]

# test data
test_img_files = []
test_dep_files = []
for i in range(5):
    test_img_file = os.path.join(dPath, "human_puzzle_{}.png".format(i))
    test_dep_file = os.path.join(dPath, "human_puzzle_data_{}.npz".format(i))
    test_img_files.append(test_img_file)
    test_dep_files.append(test_dep_file)

# BEV matrix
BEV_mat = np.load(
    os.path.join(dPath, "BEV_mat_data_0.npz"),
    allow_pickle=True
)["BEV_mat"]

# ==== [2] Prepare the bg modeler & human segmenter
bg_params = BG.Params_cv(
    history=300,
    NMixtures=5,
    varThreshold=15.,
    detectShadows=True,
    ShadowThreshold=0.55,
)
bg_extractor = BG.bgmodelGMM_cv(params=bg_params)

# human segmenter
params = hParams(det_th=20)
human_seg = Human_ColorSG_HeightInRange.buildFromImage(train_img_glove, train_depth_table, \
    intrinsic, params=params)

# puzzle segmenter
# post process - first closing for filling the holes, then opening for removing noise
kernel= np.ones((9,9), np.uint8)
mask_proc = maskproc(
    maskproc.opening, (kernel, ),
    maskproc.closing, (kernel, ),
)
puzzle_params = Puzzle_Seg.Params_Residual(
    postprocessor=lambda mask: mask_proc.apply(mask)
)
pTracker = mCentroid.centroidMulti(
    params=mCentroid.Params(plotStyle="rx")
)
puzzle_seg = Puzzle_Seg.Puzzle_Residual(theTracker=pTracker, params=puzzle_params)

# ==== [3] Learn the GMM parameters for the bg modeler
bg_extractor.doAdapt = True
ret=True
idx = 0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("The training frames and the \"Ground Truth\" foreground mask")

print("The visualization of large images is slow. So only display one example every 50 frames")
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
        bg_extractor.process(frame_train)

        # visualize
        if idx % 50 == 0:
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

# ==== [4] Test

# [4.0] utils
def postprocess_puzzle(mask):
    labels = label(mask, connectivity=1, background=0)
    plt.imshow(labels)
    plt.show()

# [4.1] get started
bg_extractor.doAdapt = False
ret=True
for idx, test_file in enumerate(test_img_files):
    test_img = cv2.imread(test_file)[:,:,::-1]
    test_dep = np.load(test_dep_files[idx], allow_pickle=True)["depth_frame"]

    # bg mask
    bg_extractor.process(test_img)
    fgMask = bg_extractor.getForeground()
    detResult = bg_extractor.getDetectResult() 
    bgMask = ~fgMask

    # human segmentation
    human_seg.update_depth(test_dep)
    human_seg.process(test_img)
    human_mask = human_seg.get_mask()

    # puzzle layer 
    puzzle_seg.set_detected_masks([bgMask, human_mask])
    puzzle_seg.process(test_img)
    puzzle_mask = puzzle_seg.get_mask()

    # BEV
    puzzle_layer = test_img * puzzle_mask[:,:,np.newaxis]
    puzzle_layer_BEV = cv2.warpPerspective(
        puzzle_layer, 
        BEV_mat, 
        (puzzle_layer.shape[1], puzzle_layer.shape[0])
    )

    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle("The GMM with Shadow detection. Test frame: {}".format(idx))

    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title("The test image")
    axes[0, 1].imshow(puzzle_mask, cmap='gray') 
    axes[0, 1].set_title("The puzzle mask")

    puzzle_seg.draw_layer(test_img, ax=axes[1,0])
    axes[1, 0].set_title("The puzzles with the MultiCentroid tracker")
    axes[1, 1].imshow(puzzle_layer_BEV)
    axes[1, 1].set_title("The puzzles from the Bird-eye-view")

plt.show()

#
#============================= puzzle01aruco =============================
