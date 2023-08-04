#!/usr/bin/python
#================================= design06trackit =================================
'''!
@brief  Use Realsense API as source stream for PuzzleScene layer processing. 

  Builds on design05 by modifying the track pointer to be the PuzzleScene one,
  which actually tracks the glove and the pieces.

Execution:
----------
Needs Intel lRealsense D435 or equivalent RGBD camera.

Just run and it displays the segmented layer image.
Hit "q" to quit.

'''
#================================= design06trackit =================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/08/04

'''
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#================================= design06trackit =================================


import numpy as np
import cv2

from Surveillance.layers.PuzzleScene import Detectors
from Surveillance.layers.PuzzleScene import TrackPointers
from Surveillance.layers.PuzzleScene import CfgPuzzleScene

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
layDet   = Detectors.load('design03saved.hdf5')
layTrack = TrackPointers() 

print('Starting ... Use "q" Quit/Move On.')

while(True):
    imageData, success = d435_starter.captureRGBD()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    layDet.detect(imageData)
    dState = layDet.getState()

    layTrack.process(dState)

    layTrack.display_cv(imageData.color, ratio=0.25, window_name="RGB")
    display.gray_cv(dState.x.astype('uint8'), ratio=0.25, window_name="Layers")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        print("Closing shop. Next: Save then Load.")
        break


quit()

#
#================================= design06trackit =================================
