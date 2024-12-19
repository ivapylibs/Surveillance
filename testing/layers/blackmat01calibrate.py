#!/usr/bin/python
#=============================== blackmat01calibrate ===============================
## @file
# @brief  Calibrate a work space consisting of a black mat. 
# 
# Builds on `design01` by encapsulating the calibration routine into a static member
# function similar to `design03calibrate`. The final output saves to an HDF5
# the calibrated system for loading prior to deployment. Only applies to the
# PuzzleScene detector.
# 
# Execution:
# ----------
# Needs Intel Realsense D435 or equivalent RGBD camera.
# 
# Run and it displays calibration instructions.
# Saves to file then quits when calibration is done.
# 
# @ingroup  TestSurveillance_Layers
# 
# @author Patricio A. Vela,   pvela@gatech.edu
# @date   2023/04/21
# 
# @quitf
#
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#=============================== blackmat01calibrate ===============================


import numpy as np
import cv2

from Surveillance.layers.PuzzleScene import Detectors

import camera.utils.display as display
import camera.d435.runner2 as d435
from camera.base import ImageRGBD



#==[0]  Get environment ready.
#
#==[0.1]    Realsense camera stream.   
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('settingsD435.yaml')
theStream = d435.D435_Runner(d435_configs)
theStream.start()


#==[0.2]    The layered detector.
#
Detectors.calibrate2config(theStream,"design03saved.hdf5")


quit()

#
#=============================== blackmat01calibrate ===============================
