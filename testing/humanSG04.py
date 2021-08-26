"""
========================================= humanSG04 ====================================

    @brief          Test the Human_colorSG_HeightInRange human segmenter
                    
                    The segmenter works the same as the testing/humanSG03.py. But the whole
                    process is wrapped in a class for more convenient usage
    
    @author         Yiye Chen,          yychen2019@gatech.edu
    @date           08/26/2021

========================================== humanSG03 =====================================
"""

# ====== [1] setup the environment. Read the data
import os
import sys
from typing import final
from PIL.Image import init
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from scipy import ndimage as ndi
import time
import copy

import trackpointer.centroid as tracker
from Surveillance.layers.human_seg import Human_ColorSG, Params
from Surveillance.utils.region_grow import RegionGrower_ValDiff as RegionGrower
from Surveillance.utils.region_grow import RG_Params
from Surveillance.utils.height_estimate import HeightEstimator

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data')

# train data. Requires two training images:
# the image of the indicator glove and the image of an empty table with the camera intrinsic parameter
train_img_glove = cv2.imread(os.path.join(dPath, 'glove_small_0.png'))[:,:,::-1]
train_depth_table = np.load(
    os.path.join(dPath, "empty_desk_data_0.npz"),
    allow_pickle=True
)["depth_frame"]
intrinsic = np.load(
    os.path.join(dPath, "empty_desk_data_0.npz"),
    allow_pickle=True
)["intrinsics"]

# ======= [2] build the segmentor instance
trackptr = tracker.centroid()
human_seg = Human_ColorSG.buildFromImage(train_img_glove, train_depth_table, intrinsic, tracker=trackptr)

# ======= [3] test on teh test image and show the result
# for each new test image, need to essentially create a new postprocess executable
timing_list = []
for i in range(6):

    # prepare visualization
    fh, axes = plt.subplots(1, 3, figsize=(15, 5))

    # read data
    rgb_path = os.path.join(dPath, "human_puzzle_small_{}.png".format(i))
    depth_path = os.path.join(dPath, "human_puzzle_small_data_{}.npz".format(i))
    test_rgb = cv2.imread(rgb_path)[:,:,::-1]
    test_depth = np.load(depth_path, allow_pickle=True)['depth_frame']

    print("Processing the image: {}".format(rgb_path))

    # process
    time_begin = time.time()
    human_seg.update_depth(test_depth)
    human_seg.process(test_rgb)
    time_end = time.time()
    timing_list.append(time_end - time_begin)

    # visualize

    plt.subplot(231)
    plt.title("The query rgb image")
    plt.imshow(test_rgb)

    plt.subplot(232)
    plt.title("The detection result from the Single-Gaussian detector")
    human_seg.draw_layer(img=test_rgb, raw_detect=True)

    plt.subplot(236)
    plt.title("The final result - after lower region removal")
    human_seg.draw_layer(img=test_rgb)

print("\n\n The average processing time for each test frame: {} sec/frame \n\n".format(np.mean(timing_list)))
plt.show()
