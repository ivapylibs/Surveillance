"""
============================================ height01 ==============================================
        @brief      Test the tabletop height reconstruction algorithm

        @author     Yiye Chen           yychen2019@gatech.edu
        @date       07/23/201
============================================ height01 ==============================================
"""

# ===== environement setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

fPath = os.path.dirname(__file__)
rPath = os.path.dirname(fPath)
uPath = os.path.join(rPath, "utils")
sys.path.append(rPath)

from utils.height_estimate import HeightEstimator 

# ===== prepare data
data_train = np.load(
    os.path.join(fPath, "data/empty_desk_data_0.npz"),
    allow_pickle=True
)
dep_train = data_train['depth_frame']
intrinsic = data_train['intrinsics']

data_test = np.load(
    os.path.join(fPath, "data/desk_hand_data_0.npz")
)
dep_test = data_test['depth_frame']
plt.imshow(dep_test)
