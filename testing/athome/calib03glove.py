#!/usr/bin/python
#================================= hand01calibrate =================================
'''!
@brief  Use Realsense API as source stream for PuzzleScene layer processing. 

  Builds on design02 by encapsulating the calibration routine into a static member
  function. The final output saves to an HDF5 the calibrated system for loading
  prior to deployment. Only applies to the PuzzleScene detector.

Execution:
----------
Needs Intel lRealsense D435 or equivalent RGBD camera.

Just run and it displays the segmented layer image.
Hit "q" to quit.

'''
#================================= hand01calibrate =================================
'''!

@author Patricio A. Vela,   pvela@gatech.edu
@date   2023/04/21

'''
#
# NOTE: Formatted for 100 column view. Using 2 space indent. Margin wrap at 8.
#
#================================= hand01calibrate =================================

#==[0] Load dependencies.
#
import numpy as np

import ivapy.display_cv as display
import camera.cv2cam as cam
from ivapy.Configuration import AlgConfig

import Surveillance.layers.GloveByColor as glove
import detector.fgmodel.Gaussian as gloveModel


#==[1]  Prep environment.
#
#==[1.1]    Realsense camera stream.   
#
cam_configs = cam.CfgColor()
cam_configs.camera.color.toRGB = True

theStream = cam.Color(cam_configs)
theStream.start()


print("hit a key when ready for snapshot")
display.wait()
rgb, good = theStream.capture()
display.rgb(rgb)
display.wait(500)

#==[1.2]    Calibrate the hand detector.
#
mu    = [220.0,50.0,100.0]
sigma = [1600.0,400.0,400.0]
Parms  = gloveModel.SGMdebug(mu    = np.array(mu),
                             sigma = np.array(sigma))
Config = gloveModel.CfgSGT.builtForRedGlove(initModel = [mu, sigma])
glove.Detector.calibrate2config(theStream, "data/glove.hdf5", initModel = [Config, Parms])


#==[1.3]    Close out and quit.
#
quit()

#
#================================= hand01calibrate =================================
