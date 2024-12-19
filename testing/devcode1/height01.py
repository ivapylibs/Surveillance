#!/usr/bin/python3
#================================ height01 ===============================
## @file
# @brief      Test the tabletop height reconstruction algorithm
# 
# @ingroup  TestSurveillance_Dev_v1
#
# @author     Yiye Chen           yychen2019@gatech.edu
# @date       2021/07/23
#
# @quitf
#
#================================ height01 ===============================

# ===== environement setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

fPath = os.path.dirname(os.path.abspath(__file__))

from Surveillance.utils.height_estimate import HeightEstimator

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

# ==== calibrate the model and apply
h_estimator = HeightEstimator(intrinsic=intrinsic)
h_estimator.calibrate(dep_train)
h_map_test = h_estimator.apply(dep_test)

# ==== display results

plt.figure()
plt.imshow(dep_train)
plt.title("The training depth frame")

plt.figure()
plt.imshow(dep_test)
plt.title("The testing depth frame")

plt.figure()
plt.imshow(h_map_test)
plt.title("The estimated height")

print("\n\n Move the cursor around the tabletop in the \'testing depth frame\' and \'The estimated height\'.")
print("Expected to see the value changes on the depth frame, whereas be almost the same on the estimated height map")

plt.show()

#
#================================ height01 ===============================
