"""
===================================== scene01_usage ========================================

    @brief          The test file demonstrate the usage of the SceneInterpreterV1

    @author         Yiye Chen.              yychen2019@gatech.edu
    @date           09/17/2021

===================================== scene01_usage ========================================
"""

# ====== [0] setup the environment.
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import detector.bgmodel.bgmodelGMM as BG
import Surveillance.layers.scene as scene
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.human_seg import Params as hParams
from Surveillance.utils.height_estimate import HeightEstimator
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg

fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/puzzle_layer')

# ==== [1] Read the data
# Run on the puzzle_layer data first to see if we can duplicate the result
# with the new class

# == [1.0] Train data

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


# train data - for the human segmenter
train_img_glove = cv2.imread(os.path.join(dPath, "calibrate_glove_0.png"))[:,:,::-1]
train_depth_table = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"), allow_pickle=True
)["depth_frame"]
train_img_table = cv2.imread(os.path.join(dPath, "empty_table_0.png"))[:,:,::-1]
intrinsic = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"), allow_pickle=True
)["intrinsics"]


# BEV matrix
BEV_mat = np.load(
    os.path.join(dPath, "BEV_mat_data_0.npz"),
    allow_pickle=True
)["BEV_mat"]



# == [1.1] Test data

# test data
test_img_files = []
test_dep_files = []
for i in range(5):
    test_img_file = os.path.join(dPath, "human_puzzle_{}.png".format(i))
    test_dep_file = os.path.join(dPath, "human_puzzle_data_{}.npz".format(i))
    test_img_files.append(test_img_file)
    test_dep_files.append(test_dep_file)


# ==== [2] Build the scene interpreter

# human segmenter - Postprocess with the dilate operation
params = hParams(
    det_th=20,
    postprocessor= lambda mask:\
        cv2.dilate(
            mask.astype(np.uint8),
            np.ones((10,10), dtype=np.uint8),
            1
        ).astype(bool)
)
human_seg = Human_ColorSG_HeightInRange.buildFromImage(train_img_glove, dep_height=None, \
    intrinsics=None, params=params)

# height estimator
height_estimator = HeightEstimator(intrinsic=intrinsic)
height_estimator.calibrate(depth_map = train_depth_table)

# bg
bg_model_params = BG.Params_cv(
    history=300,
    NMixtures=5,
    varThreshold=15.,
    detectShadows=True,
    ShadowThreshold=0.55,
)
bg_seg_params = Tabletop_Seg.Params_GMM(
    postprocessor=lambda mask: mask
)
bg_extractor = Tabletop_Seg.tabletop_GMM.build(bg_model_params, bg_seg_params)
# calibrate 
ret = True
idx = 0
human_seg.height_estimator = height_estimator   # now it needs the height_estimator. TODO: code a routine calibration process
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
human_seg.height_estimator = None   # Now no longer need that. Set to None

# robot
low_th = 0.15
high_th = 0.5
rParams = Robot_Seg.Params()
robot_seg = Robot_Seg.robot_inRange_Height(low_th=low_th, high_th=high_th, 
            theHeightEstimator=None, params=rParams)

# puzzle - postprocess with open operation
puzzle_params = Puzzle_Seg.Params_Residual(
    postprocessor=lambda mask: \
        cv2.morphologyEx(
            mask.astype(np.uint8), 
            cv2.MORPH_OPEN,
            np.ones((5,5), dtype=np.uint8)
        ).astype(bool)
)
puzzle_seg = Puzzle_Seg.Puzzle_Residual(puzzle_params)

# Scene
params = scene.Params(BEV_trans_mat=BEV_mat)
scene_interpreter = scene.SceneInterpreterV1(
    human_seg = human_seg,
    robot_seg = robot_seg,
    bg_seg=bg_extractor,
    puzzle_seg=puzzle_seg,
    heightEstimator=height_estimator,
    params=params
)


# ==== [3] Calibrate the parameters
# Left for empty for now.

# ==== [4] Test on the test data
for img_file, dep_file in zip(test_img_files, test_dep_files):
    depth = np.load(dep_file, allow_pickle=True)["depth_frame"]
    rgb = cv2.imread(img_file)[:, :, ::-1]

    scene_interpreter.process_depth(depth)
    scene_interpreter.process(rgb)

    scene_interpreter.vis_scene()

plt.show()
