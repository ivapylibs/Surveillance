#!/usr/bin/python
#================================= design06trackit =================================
## @file
# @brief  Use Realsense API as source stream for PuzzleScene layer processing. 
# 
# Builds on design05 by modifying the track pointer to be the PuzzleScene one,
# which actually tracks the glove and the pieces.
# 
# Execution:
# ----------
# Needs Intel lRealsense D435 or equivalent RGBD camera.
# 
# Just run and it displays the segmented layer image.
# Hit "q" to quit.
# 
# @ingroup  TestSurveillance_Dev
# 
# @author Patricio A. Vela,   pvela@gatech.edu
# @date   2023/08/04
# 
# @quitf
#
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#================================= design06trackit =================================


import numpy as np
import cv2

from Surveillance.layers.PuzzleScene import Detectors
from Surveillance.layers.PuzzleScene import TrackPointers
from Surveillance.layers.PuzzleScene import CfgPuzzleScene

import detector.bgmodel.onWorkspace as GWS


import camera.utils.display as display
from camera.d435.runner2 import Replay
from camera.d435.runner2 import CfgD435
from camera.base import ImageRGBD


#==[0]  Get environment ready.
#
#==[0.1]    Realsense camera stream.   
#
cfgStream = CfgD435.builtForReplay('data/20230817_161810.bag')
cfgStream.camera.align = True
theStream = Replay(cfgStream)

#==[0.2]    The layered detector.
#
layDet   = Detectors.load('puzzlebotCalib.hdf5')
layDet.workspace.bgModel.offsetThreshold(-10)
layDet.depth.state = GWS.RunState.DETECT

layTrack = TrackPointers() 

print('Starting ... Use "q" Quit/Move On.')
theStream.start()

while(True):
    imageData, success = theStream.captureRGBD()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    layDet.detect(imageData)
    dState = layDet.getState()

    layTrack.process(dState)

    layTrack.display_cv(imageData.color, ratio=0.5, window_name="RGB")
    display.gray_cv(dState.x.astype('uint8'), ratio=0.5, window_name="Layers")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        print("Closing shop. Next: Save then Load.")
        break


quit()

#
#================================= design06trackit =================================
