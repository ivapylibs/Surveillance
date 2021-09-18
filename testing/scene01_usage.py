"""
===================================== scene01_usage ========================================

    @brief          The test file demonstrate the usage of the SceneInterpreterV1

    @author         Yiye Chen.              yychen2019@gatech.edu
    @date           09/17/2021

===================================== scene01_usage ========================================
"""

# ====== [0] setup the environment.
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import detector.bgmodel.bgmodelGMM as BG
import Surveillance.layers.scene as scene
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.human_seg import Params as hParams

fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/')

# ==== [1] Read the data

# == [1.0] Train data

# == [1.1] Test data


# ==== [2] Build the scene interpreter


# ==== [3] Calibrate the parameters


# ==== [4] Test on the test data
