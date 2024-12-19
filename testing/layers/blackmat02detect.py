#!/usr/bin/python
#================================= blackmat02detect ================================
## @file
# @brief  Use Realsense API as source stream for black mat detection / object tracking. 
# 
# Builds on `design05toptrack`/`design06trackit` by having implementations
# encapsulated into a perceiver from detector and trackpointer instances.  Runs
# on camera stream.
# 
# The trackpointer for the black mat detector is nothing special at the moment. a
# multi-centroid track pointer.  It will be incrementally improved, or augmented
# through a data association strategy in the perceiver. The exact approach is TBD and
# will depend on how best to model and recognition individual puzzle pieces.
# 
# Execution:
# ----------
# Needs Intel Realsense D435 or equivalent RGBD camera.
# 
# Run and it loads calibration, then displays the segmented layer image.
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
#================================= blackmat02detect ================================


#==[0] Load dependencies.
#
import numpy as np
import cv2

import camera.utils.display as display
import camera.d435.runner2 as d435
from camera.base import ImageRGBD

import Surveillance.layers.Glove as glove
#from Surveillance.layers.Glove import Detector
#from Surveillance.layers.Glove import TrackPointer
#from Surveillance.layers.Glove import Perceiver
#from Surveillance.layers.Glove import CfgGloveTracker
#from Surveillance.layers.Glove import


#==[1]  Prep environment.
#
#==[1.1]    Realsense camera stream.   
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('settingsD435.yaml')
d435_starter = d435.D435_Runner(d435_configs)
d435_starter.start()


#==[1.2]    The layered detector.
#
gloveDet   = glove.Detector.load('design03saved.hdf5')
gloveTrack = glove.TrackPointer() 

useMethods  = glove.InstGlovePerceiver(detector=gloveDet, trackptr = gloveTrack, trackfilter = None)
cfgMethods  = None
layPerceive = glove.Perceiver(cfgMethods, useMethods)

print('Starting ... Use "q" Quit/Move On.')

while(True):
    imageData, success = d435_starter.captureRGBD()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    layPerceive.process(imageData)

    dState = layPerceive.detector.getState()

    layPerceive.tracker.display_cv(imageData.color, ratio=0.25, window_name="RGB")
    display.gray_cv(dState.x.astype('uint8'), ratio=0.25, window_name="Layers")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        print("Closing shop. Next: Save then Load.")
        break


quit()

#
#================================= blackmat02detect ================================
