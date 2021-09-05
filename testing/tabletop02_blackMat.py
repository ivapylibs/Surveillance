#!/usr/bin/python3

# ============================ tabletop02_blackMat ==============================
"""
    @brief:         Test whether a black blackground will be constrast the foreground
                    better, thereby reduce the influence of the backgrouned during 
                    the foreground substraction

                    The test will be regarding to two parts:
                    1. Difference between the foreground and the black background. Expected to be bigger
                    2. Difference between the shadow and the black background, expected to be smaller

    @author:    Yiye        yychen2019@gatech.edu
    @date:      09/05/2021
"""
# ============================ tabletop02_blackMat ==============================

# ====== [0] setup the environment. Read the data
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import detector.bgmodel.bgmodelGMM as BG
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.human_seg import Params as hParams

# ======= [0]  setup
fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/black_mat')


# ======= [1] foreground vs black/white background

# == [1.0] prepare data

# pure background data for frame difference background substraction
black_pure = cv2.imread(
    os.path.join(dPath, "BG_black_0.png")
)[:,:,::-1]
white_pure = cv2.imread(
    os.path.join(dPath, "BG_white_0.png")
)[:,:,::-1]

# prepare the 



# ======= [2] shadow vs black/white background
