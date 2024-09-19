#!/usr/bin/python
#================================== perceive01hand =================================
'''!
@brief  Simple test of depth-based hand perceiver over "flat" workspace.

  Builds on design05/design06 by having implementations encapsulated into a perceiver
  from detector and trackpointer instances.  Runs on streamed camera.


Execution:
----------
Needs Intel Realsense D435 or equivalent RGBD camera.

Just run and it displays the segmented layer image.
Hit "q" to quit.

'''
#================================== perceive01hand =================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/04/21

'''
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#
#================================== perceive01hand =================================


#==[0] Load dependencies.
#
import numpy as np
import cv2

import camera.utils.display as display
import camera.d435.runner2 as d435
from   camera.base import ImageRGBD

import Surveillance.layers.HoveringHand as hand


#==[1]  Prep environment.
#
#==[1.1]    Realsense camera stream.   
#
cam_configs = d435.CfgD435()
cam_configs.merge_from_file('blackmatD435.yaml')
theStream = d435.D435_Runner(cam_configs)
theStream.start()


#==[1.2]    The layered detector.
#
hoverDet   = hand.Detector.load('hand.hdf5')
hoverTrack = hand.TrackPointer() 

useMethods  = hand.InstPerceiver(detector=hoverDet, trackptr = hoverTrack, trackfilter = None)
cfgMethods  = None
layPerceive = hand.Perceiver(cfgMethods, useMethods)

print('Starting ... Use "q" Quit/Move On.')

while(True):
    imageData, success = theStream.captureRGBD()
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
#================================== perceive01hand =================================
