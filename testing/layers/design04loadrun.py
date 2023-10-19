#!/usr/bin/python
#================================== design02detect =================================
'''!
@brief  Use Realsense API as source stream for PuzzleScene layer processing. 

  Builds on design01 by actually running on streamed data using the default 
  configurations.  Should mostly work.  Only applies the PuzzleScene detector.


Execution:
----------
Needs Intel lRealsense D435 or equivalent RGBD camera.

Just run and it displays the segmented layer image.
Hit "q" to quit.

'''
#================================== design02detect =================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/04/21

'''
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#
#================================== design02detect =================================


import numpy as np
import cv2

from Surveillance.layers.PuzzleScene import Detectors
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
layDet = Detectors.load('design03saved.hdf5')

print('Starting ... Use "q" Quit/Move On.')

while(True):
    imageData, success = d435_starter.captureRGBD()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    layDet.detect(imageData)
    dState = layDet.getState()

    display.gray_cv(dState.x.astype('uint8'), ratio=0.25, window_name="Layers")
    display.rgb_cv(imageData.color, ratio=0.25, window_name="RGB")
    #display.binary_cv(dState.x, window_name="RGB")
    #display.rgb_depth_cv(imageData.color, imageData.depth, ratio=0.25, window_name="RGB")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        print("Closing shop. Next: Save then Load.")
        break


quit()

#
#================================== design02detect =================================
