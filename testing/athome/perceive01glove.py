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


#==[0] Load dependencies.
#
import numpy as np
import cv2

import ivapy.display_cv as display
import camera.cv2cam as cam

import Surveillance.layers.GloveByColor as glove


#==[1]  Prep environment.
#
#==[1.1]    Realsense camera stream.   
#
cam_configs = cam.CfgColor()
cam_configs.camera.color.toRGB = True

theStream = cam.Color(cam_configs)
theStream.start()


#==[1.2]    The glove detector.
#
gloveDet   = glove.Detector.load('data/glove.hdf5')
gloveTrack = glove.TrackPointer() 

useMethods  = glove.InstGlovePerceiver(detector=gloveDet, trackptr = gloveTrack, trackfilter = None)
cfgMethods  = None
layPerceive = glove.Perceiver(cfgMethods, useMethods)

print('Starting ... Use "q" Quit/Move On.')

while(True):
    image, success = theStream.capture()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    layPerceive.process(image)

    dState = layPerceive.detector.getState()

    layPerceive.tracker.display_cv(image, ratio=0.25, window_name="RGB")
    display.gray(dState.x.astype('uint8'), ratio=0.25, window_name="Layers")

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        print("Closing shop. Next: Save then Load.")
        break


quit()

#
#================================== design02detect =================================
