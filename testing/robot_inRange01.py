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
empty_table_depth = None
intrinsic = None 

# test data
robot_rgb = None
robot_depth = None

# ===== [2] prepare the segmenter
height_estimator = HeightEstimator()
height_estimator.calibrate(empty_table_depth)

low_th = 0.05
high_th = 50
# treat the height_estimation as the preprocessor
params = Params(preprocessor=lambda depth:height_estimator.apply(depth))

robot_seg = robot_inRange(low_th=low_th, high_th=high_th, params=params)

# ======= [3] test on teh test image and show the result
# for each new test image, need to essentially create a new postprocess executable
# Seems not that elegent
robot_seg.process(robot_depth)

plt.figure()
robot_seg.draw_layer(img=robot_rgb)
plt.show()