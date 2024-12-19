#!/usr/bin/python
#================================= design05toptrack ================================
## @file
# @brief  Use Realsense API as source stream for PuzzleScene layer processing. 
# 
#   Used for testing and development of the glove track pointer.  Applies to
#   the PuzzleScene implementation.  Builds on `design03calibrate` by running
#   on streamed data using the default configurations.  Should mostly work.  
# 
# Execution:
# ----------
# Needs Intel Realsense D435 or equivalent RGBD camera.
# 
# Run and it load calibration, then processing image with emphasis on
# displaying the glove track point.  
#
# Hit "q" to quit.
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
#================================= design05toptrack ================================


import numpy as np
import cv2

from Surveillance.layers.PuzzleScene import Detectors
from Surveillance.layers.PuzzleScene import CfgPuzzleScene
import trackpointer.toplines as tp

import camera.utils.display as display
import camera.d435.runner2 as d435
from camera.base import ImageRGBD



#==[0]  Get environment ready.
#
#==[0.1]    Realsense camera stream.   
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('settingsD435.yaml')
d435_starter = d435.D435_Runner(d435_configs)
d435_starter.start()


#==[0.2]    The layered detector.
#
layDet = Detectors.load('design03saved.hdf5')
tpGlove = tp.fromBottom()

print('Starting ... Use "q" Quit/Move On.')

while(True):
    imageData, success = d435_starter.captureRGBD()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    layDet.detect(imageData)
    dState = layDet.getState()

    tpGlove.process(layDet.imGlove)

    if (tpGlove.haveMeas):
      display.trackpoint_cv(imageData.color, tpGlove.tpt, ratio=0.25, window_name="RGB")
    else:
      display.rgb_cv(imageData.color, ratio=0.25, window_name="RGB")

    display.gray_cv(dState.x.astype('uint8'), ratio=0.25, window_name="Layers")
    #display.binary_cv(dState.x, window_name="RGB")
    #display.rgb_depth_cv(imageData.color, imageData.depth, ratio=0.25, window_name="RGB")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        print("Closing shop. Next: Save then Load.")
        break


quit()

#
#================================= design05toptrack ================================
