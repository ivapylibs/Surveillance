#!/user/bin/python
#=============================== humanSG03 ===============================
## @file
# @brief    Test the single-Gaussian-based human segmentor with the detector,
#           the tracker, and the threshold + connected component analysis
#           Train(calibrate) and test on single frames instead of videos. Human
#           hand + puzzle pieces.
# 
# By replacing the region grow in the humanSG02 with the hysteresis, hope to
# address the algorithm speed and the variance explode issue (these two are
# often associated) The lower threshold is set to the minimum value of the
# color detected pixels (with conservative threshold and high precision), and
# the higher threshold as the mean value.
#
# @ingroup  TestSurveillance_Dev_v1
#
# @author         Yiye Chen,          yychen2019@gatech.edu
# @date           2021/08/05
# 
# @quitf
#
#=============================== humanSG03 ===============================

# ====== [1] setup the environment. Read the data
import os
import sys
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
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
height_estimator = HeightEstimator(intrinsic)
height_estimator.calibrate(train_depth_table)

def post_process(depth, init_mask, fh=None):
    """
    The function to:
    (a) get the height map from the depth map
    (b) perform thresholding on the height map and find the connected component to the largest CC of the init_mask
    (c) assuming the hand is reaching out from the top of the image frame, remove all pixels so far below the init_mask as outlier
    """

    # threshold
    height_map = np.abs(height_estimator.apply(depth))
    init_height = height_map[init_mask]
    low = np.amin(init_height)
    mask = height_map > low 

    # Connected components of mask 
    labels_mask, num_labels = ndi.label(mask)
    # Check which connected components contain pixels from mask_high.
    sums = ndi.sum(init_mask, labels_mask, np.arange(num_labels + 1))
    connected_to_max_init = sums == max(sums)   # by take only the max, the non-largest connected component of the init_mask will be ignored
    max_connect_mask = connected_to_max_init[labels_mask]

    # remove pixels so far below the init mask
    cols_init = np.where(init_mask==1)[0]
    col_max = np.amax(cols_init)
    final_mask = copy.deepcopy(max_connect_mask)
    final_mask[col_max+10:] = 0

    print("The lowest detected height: {}. ".format(low))

    # visualization
    if fh is None:
        f = plt.gcf()
    else:
        f = fh

    f.add_subplot(233)
    plt.title("the estimated height map")
    plt.imshow(height_map)

    f.add_subplot(234)
    plt.title("Areas with height larger than the minimum hand height")
    plt.imshow(mask, cmap='gray')

    f.add_subplot(235)
    plt.title("The area connected to the detection mask")
    plt.imshow(max_connect_mask, cmap='gray')

    return final_mask 

trackptr = tracker.centroid()
human_seg = Human_ColorSG.buildFromImage(train_img_glove, trackptr)

# ======= [3] test on teh test image and show the result
# for each new test image, need to essentially create a new postprocess executable

timing_list = []
for i in range(6):

    # prepare visualization
    fh, axes = plt.subplots(2, 3, figsize=(15, 10))

    # read data
    rgb_path = os.path.join(dPath, "human_puzzle_small_{}.png".format(i))
    depth_path = os.path.join(dPath, "human_puzzle_small_data_{}.npz".format(i))
    test_rgb = cv2.imread(rgb_path)[:,:,::-1]
    test_depth = np.load(depth_path, allow_pickle=True)['depth_frame']

    print("Processing the image: {}".format(rgb_path))

    # process
    time_begin = time.time()
    postP = lambda init_mask: post_process(test_depth, init_mask, fh=fh)
    human_seg.update_params("postprocessor", postP)
    human_seg.process(test_rgb)
    time_end = time.time()
    timing_list.append(time_end - time_begin)

    # visualize

    plt.subplot(231)
    plt.title("The query rgb image")
    plt.imshow(test_rgb)

    #plt.subplot(232)
    #plt.title("The query depth image")
    #plt.imshow(test_depth)

    plt.subplot(232)
    plt.title("The detection result from the Single-Gaussian detector")
    human_seg.draw_layer(img=test_rgb, raw_detect=True)

    plt.subplot(236)
    plt.title("The final result - after lower region removal")
    human_seg.draw_layer(img=test_rgb)

print("\n\n The average processing time for each test frame: {} sec/frame \n\n".format(np.mean(timing_list)))
plt.show()


#
#=============================== humanSG03 ===============================
