"""

    @brief:     The test script for the coordinate transformation using the d435 camera, aruco-based world frame, and the manually set world-to-robot transformation

    @author:    Yiye Chen.      yychen2019@gatech.edu
    @date:      11/28/2021

"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

import camera.d435.runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
import camera.utils.display as display

from Surveillance.utils.transform import frameTransformer

# The world-to-robot transformation
from Surveillance.utils.transform_mats import M_WtoR
from coordTransform_colors import *

# The D435 starter
d435_configs = d435.D435_Configs(
    W_dep=848,
    H_dep=480,
    W_color=1920,
    H_color=1080
)

d435_starter = d435.D435_Runner(d435_configs)

# The aruco calibrator
calibrator_CtoW = CtoW_Calibrator_aruco(
    d435_starter.intrinsic_mat,
    distCoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    markerLength_CL = 0.078,
    maxFrames = 30,
    flag_vis_extrinsic = True,
    flag_print_MCL = False,
    stabilize_version =False 
)

# the frame Transformer
frame_transformer = frameTransformer(
    M_intrinsic=d435_starter.intrinsic_mat,
    M_WtoC=None,
    M_WtoR=M_WtoR,
)

# sample some target points from the robot frame
target_points_R = np.array([
    [0.2, -0.15, 0],
    [0.2, 0, 0],
    [0.2, 0.15, 0],
    [0.3, -0.15, 0],
    [0.3, 0, 0],
    [0.3, 0.15, 0],
])


plt.figure(1)
plt.show(block=False)
plt.ion()
cali_finish = False
while(not cali_finish):
    # get frames
    rgb, dep, success = d435_starter.get_frames()
    if not success:
        print("Cannot get the camera signals. Exiting...")
        exit()

    # calibrate
    M_CL, corners_aruco, img_with_ext, status = calibrator_CtoW.process(rgb, dep) 

    # update M_CtoW
    frame_transformer.M_WtoC = M_CL

    # update the flag
    cali_finish = calibrator_CtoW.stable_status

    # visualize calibration
    display.display_images_cv([img_with_ext[:,:,::-1]], ratio=0.5, \
        window_name="The extrinsic calibration(right, Press \'q\' to exit") 

    
    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break


    # transform the target points and visualize them
    p_W, p_C, p_Img = frame_transformer.parsePRob(target_points_R)
    print("The world frame coordinate: {}".format(p_W))
    print("The camera frame coordinate: {}".format(p_C))
    print("The image frame coordinate: {}".format(p_Img))

    # plot the target points
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    for tP in p_Img.astype(int):
        rgb = cv2.circle(rgb, (tP[0], tP[1]), radius=5, color=color_imgF_orig, thickness=-1)
    display.display_images_cv([rgb[:,:,::-1]], ratio=0.5, \
        window_name="The projection of the target position Press \'q\' to exit") 



cv2.destroyAllWindows()
print("Calibration complete")

# parse the target points
p_W, p_C, p_Img = frame_transformer.parsePRob(target_points_R)
print("The world frame coordinate: {}".format(p_W))
print("The camera frame coordinate: {}".format(p_C))
print("The image frame coordinate: {}".format(p_Img))

while(True):

    # get frames
    rgb, dep, success = d435_starter.get_frames()

    #pC_map, pW_map, pR_map = frame_transformer.parseDepth(dep)
    #plt.figure(1)
    #plt.imshow(pW_map[:,:,-1])
    #plt.draw()
    #plt.pause(1)

    # plot the target points
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    for tP in p_Img.astype(int):
        rgb = cv2.circle(rgb, (tP[0], tP[1]), radius=5, color=color_imgF_orig, thickness=-1)
    display.display_images_cv([rgb[:,:,::-1]], ratio=0.5, \
        window_name="The projection of the target position Press \'q\' to exit") 

    opKey = cv2.waitKey(1)
    if opKey == ord('q'):
        break
