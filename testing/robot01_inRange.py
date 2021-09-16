"""
========================================= robot_inRange01 ====================================

    @brief          Test the in-range-detection-based robot layer segmenter 

    @author         Yiye Chen,          yychen2019@gatech.edu
    @date           07/29/2021

========================================== robot_inRange01 =====================================
"""

# ====== [1] setup the environment. Read the data
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import trackpointer.centroid as tracker
from Surveillance.layers.robot_seg import robot_inRange, Params
from Surveillance.utils.region_grow import RegionGrower
from Surveillance.utils.height_estimate import HeightEstimator

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data')

# train data
data = np.load(
    os.path.join(dPath, "empty_desk_data_0.npz"),
    allow_pickle=True
)
empty_table_depth = data['depth_frame']
intrinsic = data['intrinsics'] 

# test data
robot_rgbs = []
robot_depths = []
for i in range(3):
    robot_rgbs.append(
        plt.imread(os.path.join(dPath, "robot_small_{}.png".format(i)))
    )
    robot_depths.append(
        np.load(
            os.path.join(dPath, "robot_small_data_{}.npz".format(i))
        )['depth_frame']
    )

# ===== [2] prepare the segmenter
height_estimator = HeightEstimator(intrinsic=intrinsic)
height_estimator.calibrate(empty_table_depth)

low_th = 0.15
high_th = 0.5
# treat the height_estimation as the preprocessor. Take absolute value
params = Params(preprocessor=lambda depth:\
    np.abs(height_estimator.apply(depth))
)

robot_seg = robot_inRange(low_th=low_th, high_th=high_th, params=params)

# ======= [3] test on teh test image and show the result
# for each new test image, need to essentially create a new postprocess executable
# Seems not that elegent

for robot_rgb, robot_depth in zip(robot_rgbs, robot_depths):
    robot_seg.process(robot_depth)

    fig,axes = plt.subplots(1,2, figsize=(12, 6))
    fig.tight_layout()
    plt.subplot(121)
    plt.title("The rgb image")
    plt.imshow(robot_rgb)
    plt.subplot(122)
    plt.title("The robot layer segmentation")
    robot_seg.draw_layer(img=robot_rgb)

plt.show()