#!/usr/bin/python
#================================ inimage04calibrate ===============================
"""!
@brief  Test out calibration routine that gets user input and saves regions.

This code tests the case that an initial region mask is available.  In this case,
it provides one pre-existing region. The calibration routine then adds additional
regions based on the user input.

The version without initial regions was manually tested and works. The conditional
logic is correct.

"""
#================================ inimage04calibrate ===============================
"""!
@file       inimage04calibrate.py

@author     Patricio A. Vela,       pvela@gatech.edu
@date       2023/12/21
"""

#
# NOTE: 90 columns, 2 space indent, wrap margin 5.
#
#================================ inimage04calibrate ===============================


#==[0] Environment setup.
#
import detector.activity.byRegion as regact
import numpy as np
import ivapy.display_cv as display
import camera.cv2cam as cam



#==[1] Testing/demonstration code to isntantiate and specify activity regions.
#
camfig = cam.CfgColor()
camfig.camera.color.toRGB = True

theStream = cam.Color(camfig)
theStream.start()

theImage, success = theStream.capture()

theStream.stop()

print("[1] Construct based on specified image, initialized region, and annotations.")
print("    Saving to file.")
regact.imageRegions.calibrateFromPolygonMouseInputOverImageRGB(\
                                                         theImage,'data/regions.hdf')


print("[2] Load from file and apply.")
theActivity = regact.imageRegions.load('data/regions.hdf')

theActivity.display_cv(window_name = "Loaded")

print("    Sending signals and testing activity states. Outcome depends on user input.");
theActivity.process([[5],[5]])
print(theActivity.x)
theActivity.process([[50],[50]])
print(theActivity.x)
theActivity.process([[200],[75]])
print(theActivity.x)
theActivity.process([[300],[75]])
print(theActivity.x)

display.wait()

#
#================================ inimage04calibrate ===============================
