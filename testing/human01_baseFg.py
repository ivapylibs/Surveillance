"""
=========================== human_humanSG02 ======================

    @brief          Test the single-Gaussian-based human segmentor 
                    with only the detector and the tracker

=========================== human_humanSG01 ======================
"""

# ====== [1] setup the environment. Read the data
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import trackpointer.centroid as tracker
from Surveillance.layers.human_seg import Human_ColorSG

fPath = os.path.realpath(__file__)
tPath = os.path.dirname(fPath)
dPath = os.path.join(tPath, 'data')

img_train = cv2.imread(
    os.path.join(dPath, "glove_1.png")
)[:, :, ::-1]


img_test = cv2.imread(
    os.path.join(dPath, "glove_2.png")
)[:, :, ::-1]

# ======= [2] build the segmentor instance
trackptr = tracker.centroid()
human_seg = Human_ColorSG.buildFromImage(img_train, trackptr)

# ======= [3] test on teh test image and show the result
human_seg.process(img_test)

plt.figure()
human_seg.draw_layer(img=img_test)
plt.show()