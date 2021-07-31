"""
========================================= humanSG02 ====================================

    @brief          Test the single-Gaussian-based human segmentor 
                    with the detector, the tracker, and the depth-region-grow
                    as the postprocess.
                    Train(calibrate) and test on single frames instead of videos
    
    @author         Yiye Chen,          yychen2019@gatech.edu
    @date           07/29/2021

========================================== humanSG02 =====================================
"""

# ====== [1] setup the environment. Read the data
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import trackpointer.centroid as tracker
from Surveillance.layers.human_seg import Human_ColorSG
from Surveillance.utils.region_grow import RegionGrower
from Surveillance.utils.height_estimate import HeightEstimator

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data')

# train data. Requires two training images:
# the image of the indicator glove and the image of an empty table with the camera intrinsic parameter
train_img_glove = None
train_img_table = None
intrinsic = None

# test data
test_rgb = None
test_depth = None

# ======= [2] build the segmentor instance
height_estimator = HeightEstimator(intrinsic)
height_estimator.calibrate(train_img_table)
region_grower = RegionGrower()
def post_process(depth, init_mask):
    """
    The function get the height map from the depth map, and start the region grow from the init_mask
    """
    height_map = height_estimator.apply(depth)
    region_grower.process_mask(height_map, init_mask)
    return region_grower.get_final_mask()

trackptr = tracker.centroid()
human_seg = Human_ColorSG.buildFromImage(train_img_glove, trackptr, post_process=None)

# ======= [3] test on teh test image and show the result
# for each new test image, need to essentially create a new postprocess executable
# Seems not that elegent
postP = lambda init_mask: post_process(test_depth, init_mask)
human_seg.params.postprocessor = postP
human_seg.update_params("postprocessor", postP)
human_seg.process(test_rgb)

plt.figure()
human_seg.draw_layer(img=test_rgb)
plt.show()