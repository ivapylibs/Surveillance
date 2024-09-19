#!/usr/bin/python
#================================= hand01calibrate =================================
'''!
@brief  Use Realsense API as source stream for calibrating the tabletop/workspace
        depth model.  Deviations from this (towards the camera) are considered 
        part of a hovering hand in the workspace.

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
import camera.utils.display as display
import camera.d435.runner2  as d435

import Surveillance.layers.HoveringHand as hand

#==[1]  Prep environment.
#
#==[1.1]    Realsense camera stream.   
#
cam_configs = d435.CfgD435()
cam_configs.merge_from_file('blackmatD435.yaml')
theStream = d435.D435_Runner(cam_configs)
theStream.start()


#==[1.2]    Calibrate the hand detector.
#
hand.Detector.calibrate2config(theStream,"hand.hdf5")


#==[1.3]    Close out and quit.
#
quit()

#
#================================= hand01calibrate =================================
