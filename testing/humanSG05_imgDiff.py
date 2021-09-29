"""
========================================= humanSG04 ====================================

    @brief          Test the Human_colorSG_HeightInRange human segmenter with the automatic
                    img difference builder

                    Comparing to the testing/humanSG04.py, except for the builder, the other 
                    difference is the testing data is collected under the upgraded light condition
                    
    @author         Yiye Chen,          yychen2019@gatech.edu
    @date           09/26/2021

========================================== humanSG03 =====================================
"""

# ====== [1] setup the environment. Read the data
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np

import trackpointer.centroid as tracker
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange, Params

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data/puzzle_layer')

# train data.
empty_table_rgb = cv2.imread(os.path.join(dPath, "empty_table_0.png"))[:,:,::-1]
empty_table_depth = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"),
    allow_pickle=True
)["depth_frame"]
glove_rgb = cv2.imread(os.path.join(dPath, "calibrate_glove_0.png"))[:,:,::-1]
intrinsic = np.load(
    os.path.join(dPath, "empty_table_data_0.npz"),
    allow_pickle=True
)["intrinsics"]

# ======= [2] build the segmentor instance
trackptr = tracker.centroid()
human_seg = Human_ColorSG_HeightInRange.buildImgDiff(
    empty_table_rgb, 
    glove_rgb, 
    empty_table_depth, 
    intrinsic, 
    tracker=trackptr, 
    params=Params(
        det_th=8
    )
)

# ======= [3] test on teh test image and show the result
# for each new test image, need to essentially create a new postprocess executable
timing_list = []
for i in range(5):

    # prepare visualization
    fh, axes = plt.subplots(1, 3, figsize=(15, 5))

    # read data
    rgb_path = os.path.join(dPath, "human_puzzle_{}.png".format(i))
    depth_path = os.path.join(dPath, "human_puzzle_data_{}.npz".format(i))
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
    plt.subplot(221)
    plt.title("The query rgb image")
    plt.imshow(test_rgb)

    plt.subplot(222)
    plt.title("The estimated height map")
    plt.imshow(human_seg.height_map)

    plt.subplot(223)
    plt.title("The detection result from the Single-Gaussian detector")
    human_seg.draw_layer(img=test_rgb, raw_detect=True)

    plt.subplot(224)
    plt.title("The final result - after lower region removal")
    human_seg.draw_layer(img=test_rgb)


print("\n\n The average processing time for each test frame: {} sec/frame \n\n".format(np.mean(timing_list)))
plt.show()
