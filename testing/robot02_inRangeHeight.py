"""
========================================= robot02_inRangeHeight ====================================

    @brief          Test the Height-based inRange robot segmenter class

                    Comparing to the robot01_inRange.py, the class tested here follow the 
                    other segmenters that the process function is for the rgb image.
                    A seperate process_depth for the depth is created, in which the height estimated
                    The inRange detection on height is moved to the postprocess part because 
                    it is not rgb-related

    @author         Yiye Chen,          yychen2019@gatech.edu
    @date           09/18/2021

========================================== robot02_inRangeHeight =====================================
"""

# ====== [1] setup the environment. Read the data
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import trackpointer.centroid as tracker
from Surveillance.layers.robot_seg import robot_inRange_Height, Params
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
# all required is programmed in the routine post-process of the class, so nothing required here
params = Params()

robot_seg = robot_inRange_Height(low_th=low_th, high_th=high_th, 
            theHeightEstimator=height_estimator, params=params)

# ======= [3] test on teh test image and show the result
# for each new test image, need to essentially create a new postprocess executable
# Seems not that elegent

for robot_rgb, robot_depth in zip(robot_rgbs, robot_depths):
    robot_seg.process_depth(robot_depth)
    robot_seg.process(robot_rgb)

    fig,axes = plt.subplots(1,2, figsize=(12, 6))
    fig.tight_layout()
    plt.subplot(121)
    plt.title("The rgb image")
    plt.imshow(robot_rgb)
    plt.subplot(122)
    plt.title("The robot layer segmentation")
    robot_seg.draw_layer(img=robot_rgb)

plt.show()