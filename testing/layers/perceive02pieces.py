#!/usr/bin/python
#================================= perceive02pieces ================================
## @file
# @brief  Use Realsense API as source stream for puzzle piece tracking. 
# 
# Builds out a basic puzzle piece perceiver, which consists of a calibrated detector
# and the simplest puzzle piece trackpointer: a multi-centroid track pointer.  Later
# versions should improve this to model puzzle pieces more richly and to incorporate
# association via a track filter.
# 
# The puzze piece centroid track settings have a generic minimum and maximum area that
# should ignore the hand as it comes in and moves pieces around, except when the hand
# initially enters and has a small area.  Also ignores any pieces connected to the 
# hand+arm combo.
# 
# In principle would also apply to the case that a manipulator is in the scene.
# 
# Execution:
# ----------
# Needs Intel Realsense D435 or equivalent RGBD camera.
# 
# Run and it loads stored calibration, then displays the puzzle piece track
# points (no data association).
# Hit "q" to quit.
# 
# @ingroup  TestSurveillance_Layers
#
# @author Patricio A. Vela,   pvela@gatech.edu
# @date   2023/12/08
# 
# @quitf
#
# NOTE: Formatted for 100 column view. Using 4 space indent.
#
#================================= perceive02pieces ================================


#==[0] Load dependencies.
#
import camera.utils.display as display
import camera.d435.runner2  as d435

import Surveillance.layers.BlackWorkMat as blackmat


#==[1]  Prep environment.
#
#==[1.1]    Realsense camera stream.   
#
camStream = d435.D435_Runner.buildFromFile('blackmatD435.yaml')
camStream.start()


#==[1.2]    The black mat detector and puzzle piece perceiver.
#
detConfig = blackmat.defaultBuildCfg_DetectorLoad('blackmat.hdf5')
perceiver = blackmat.PuzzlePerceiver.buildWithBasicTracker(detConfig)

print('Starting ... Use "q" Quit/Move On.')

while(True):
    imageData, success = camStream.captureRGBD()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    perceiver.process(imageData.color)

    #dState = perceiver.detector.getState()

    perceiver.display_cv(imageData.color, ratio=0.5)

    opKey = display.wait_cv(1)
    if opKey == ord('q'):
        print("Closing shop. Next: Save then Load.")
        break


quit()

#
#================================= perceive02pieces ================================
