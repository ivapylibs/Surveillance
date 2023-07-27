#!/usr/bin/python
#================================ design03calibrate ================================
'''!
@brief  Use Realsense API as source stream for PuzzleScene layer processing. 

  Builds on design02 by encapsulating the calibration routine into a static member
  function. The final output saves to an HDF5 the calibrated system for loading
  prior to deployment. Only applies to the PuzzleScene detector.

Execution:
----------
Needs Intel lRealsense D435 or equivalent RGBD camera.

Just run and it displays the segmented layer image.
Hit "q" to quit.

'''
#================================ design03calibrate ================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/04/21

'''
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#
#================================ design03calibrate ================================


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
#================================ design03calibrate ================================
