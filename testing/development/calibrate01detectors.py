#!/usr/bin/python
#=============================== calibrate01detectors ==============================
'''!
@brief  Use Realsense API as source stream for PuzzleScene detector calibration.

  Taken from "layers/design03calibrate" and adjusted slightly based on intended
  use.  Some changes are going to be in the settings files.

  Only applies to the PuzzleScene detector.  Later calibration routines will 
  explore the fuller calibration process.  It will probably take a minute or
  so in the end.

Execution:
----------
Needs Intel Realsense D435 or equivalent RGBD camera.

Just run. Follow instructions/explanation provided.
Hit "q" to quit.

'''
#=============================== calibrate01detectors ==============================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/08/18

'''
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#
#=============================== calibrate01detectors ==============================


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
d435_configs.merge_from_file('puzzlebotD435.yaml')
theStream = d435.D435_Runner(d435_configs)
theStream.start()


#==[0.2]    The layered detector.
#

Detectors.calibrate2config(theStream,"puzzlebotCalib.hdf5")

quit()

#
#=============================== calibrate01detectors ==============================
