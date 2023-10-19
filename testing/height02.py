#!/usr/bin/python
#===================================== height02 ====================================
'''!
@brief  Use Realsense API as source stream for workspace planar height processing. 

  Builds on height01 by actually running on streamed data using the default 
  configurations.  Should mostly work.  

Execution:
----------
Needs Intel lRealsense D435 or equivalent RGBD camera.

Just run and it displays the height image.
Hit "q" to quit.

'''
#===================================== height02 ====================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/04/21

'''
# NOTE: Formatted for 100 column view. Using 2 space indent.
#
#
#===================================== height02 ====================================


import numpy as np
import cv2

from Surveillance.utils.height_estimate import HeightEstimator

import camera.utils.display as display
import camera.d435.runner2 as d435
from camera.base import ImageRGBD

import matplotlib.pyplot as plt

#==[0]  Get environment ready.
#
#==[0.1]    Realsense camera stream.   
#
d435_configs = d435.CfgD435()
d435_configs.merge_from_file('settingsD435.yaml')
d435_starter = d435.D435_Runner(d435_configs)
d435_starter.start()


#==[0.2]    The height estimator.
#
iMat = np.array([[1.38106177e+03, 0.00000000e+00, 9.78223145e+02],
                 [0.00000000e+00, 1.38116895e+03, 5.45521362e+02],
                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
h_estimator = HeightEstimator(intrinsic=iMat)

plt.figure()

print('Starting ... Use "q" Quit/Move On.')

print('Getting image for depth plane calibration ...')


imageData, success = d435_starter.captureRGBD()
if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

h_estimator.calibrate(imageData.depth)

hSmooth = None
salpha  = 0.9

while(1):
  imageData, success = d435_starter.captureRGBD()

  h_map_test = h_estimator.apply(imageData.depth)

  if (hSmooth is None):
    hSmooth = h_map_test
  else:
    hSmooth = (1-salpha)*hSmooth + salpha*h_map_test

  tooHigh = np.greater(hSmooth,0.015)

  #-- Clip depth and height data for good scaling over values that matter.
  #
  depth = np.minimum(np.maximum(imageData.depth, 0.4),1.2)
  h_map_test = np.add(np.minimum(np.maximum(h_map_test , -0.05), 0.5), 0.05)

  #-- Display color, depth, height, and binary mask images.
  #
  display.rgb_cv(imageData.color, ratio=0.25, window_name="Color")
  display.depth_cv(h_map_test, depth_clip=0.0, ratio=0.25, window_name="Height")
  display.depth_cv(depth, depth_clip=0.0, ratio=0.25, window_name="Depth")
  display.binary_cv(tooHigh, ratio=0.25, window_name="TooHigh")

  opKey = cv2.waitKey(1)
  if opKey == ord('q'):
      print("Closing shop. Next: Save then Load.")
      break


quit()

#
#===================================== height02 ====================================
