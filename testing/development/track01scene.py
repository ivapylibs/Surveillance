#!/usr/bin/python
#=================================== track01scene ==================================
'''!
@brief  Use Realsense API as source stream for PuzzleScene layer processing. 


Taken from "layers/deisgn06trackit" and modifies by integrating Perceiver instance
without track filters.  Will slowly reflect full Perceiver integration.  Right
now that is not the case.

Execution:
----------
Needs Intel Realsense D435 or equivalent RGBD camera.

Just run and it displays the segmented layer image.
Hit "q" to quit.

'''
#=================================== track01scene ==================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/08/18

'''
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#=================================== track01scene ==================================


import numpy as np
import cv2

import Surveillance.layers.PuzzleScene as Scene

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

#==[0.2]    The layered puzzle scene perceiver.
#
layDet   = Scene.Detectors.load('puzzlebotCalib.hdf5')
layTrack = Scene.TrackPointers() 
layFilt  = None

perInst  = Scene.InstPuzzlePerceiver(detector = layDet, trackptr = layTrack, trackfilter = layFilt)
layPer   = Scene.Perceiver(None, perInst)

print('Starting ... Use "q" Quit/Move On.')

while(True):
    imageData, success = theStream.captureRGBD()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    layPer.process(imageData)

    layPer.tracker.display_cv(imageData.color, ratio=0.5, window_name="RGB")

    dState = layPer.detector.getState()
    display.gray_cv(dState.x.astype('uint8'), ratio=0.5, window_name="Layers")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        print("Closing shop.")
        break


quit()

#
#=================================== track01scene ==================================
