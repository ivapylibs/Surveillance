#!/usr/bin/python
#================================= perceive01glove =================================
'''!
@brief  Use Realsense API as source stream for PuzzleScene tracking. 

  Builds on design05/design06 by having implementations encapsulated into a perceiver
  from detector and trackpointer instances.  Runs on camera streamed.

  At the date of creation, the puzzle piece trackpointer is just a multi-centroid 
  track pointer.  It will be incrementally improved, or augmented through a
  data association strategy in the perceiver. The exact approach is TBD and will
  depend on how best to model and recognition individual puzzle pieces.

Execution:
----------
Needs Intel Realsense D435 or equivalent RGBD camera.

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
from Surveillance.layers.PuzzleScene import TrackPointers
from Surveillance.layers.PuzzleScene import Perceivers
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
layTrack = TrackPointers() 

layPerceive = Perceiver(layDet, layTrack)

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
#================================== design02detect =================================
