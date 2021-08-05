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
region_grower = RegionGrower(RG_Params(
    k_neigb=8,
    th_val=0.03,
    th_var=10.
))
def post_process(depth, init_mask):
    """
    The function get the height map from the depth map, and start the region grow from the init_mask
    """
    height_map = height_estimator.apply(depth)
    noH_mask = (np.abs(height_map) < 0.1)
    plt.figure()
    plt.subplot(121)
    plt.title("Height map")
    plt.imshow(np.abs(height_map))
    plt.subplot(122)
    plt.title("Estimated non-human mask from the height")
    plt.imshow(noH_mask, cmap='gray')
    region_grower.process_mask(height_map, init_mask, noH_mask)
    return region_grower.get_final_mask()

trackptr = tracker.centroid()
human_seg = Human_ColorSG.buildFromImage(train_img_glove, trackptr)

# ======= [3] test on teh test image and show the result
# for each new test image, need to essentially create a new postprocess executable

for i in range(2):

    # read data
    rgb_path = os.path.join(dPath, "human_puzzle_small_{}.png".format(i))
    depth_path = os.path.join(dPath, "human_puzzle_small_data_{}.npz".format(i))
    test_rgb = cv2.imread(rgb_path)[:,:,::-1]
    test_depth = np.load(depth_path, allow_pickle=True)['depth_frame']

    print("Processing the image: {}".format(rgb_path))


    # process
    postP = lambda init_mask: post_process(test_depth, init_mask)
    human_seg.update_params("postprocessor", postP)
    human_seg.process(test_rgb)

    # visualize
    plt.subplots(1, 3, figsize=(15, 5))

    plt.subplot(131)
    plt.title("The query image")
    plt.imshow(test_rgb)
    plt.subplot(132)
    plt.title("The detection result from the Single-Gaussian detector")
    human_seg.draw_layer(img=test_rgb, raw_detect=True)
    plt.subplot(133)
    plt.title("After the height-region grow post-process")
    human_seg.draw_layer(img=test_rgb)


plt.show()