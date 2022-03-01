"""

    @brief:     The rosbag data recorder records the calibration data and test data into a single rosbag.
                It won't run the system on the test data

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           02/24/2022

"""

import numpy as np
import os
import sys
import subprocess
import time

import rospy
import rosgraph

from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams
from Surveillance.deployment.utils import terminate_process_and_children

if __name__ == "__main__":

    # start the roscore if necessary
    roscore_proc = None
    if not rosgraph.is_master_online():
        roscore_proc = subprocess.Popen(['roscore'])
        # wait for a second to start completely
        time.sleep(1)
    
    # init core
    rospy.init_node("Surveillance_Data_Recorder")

    # == [0] Start the rosbag recording 
    rosbag_name = "data.bag"
    command = "rosbag record -a -o {}".format(rosbag_name)
    rosbag_proc = subprocess.Popen(command, shell=True)
    time.sleep(5)

    # == [1] Build
    configs = bParams(
        markerLength = 0.08,
        W = 1920,               # The width of the frames
        H = 1080,                # The depth of the frames
        reCalibrate = True,
        ros_pub = True,         # Publish the test data to the ros or not
        test_rgb_topic = "test_rgb",
        test_depth_topic = "test_depth",
        visualize = True,
        run_system=False        # Only save, don't run
    )
    data_collector = BaseSurveillanceDeploy.buildPub(configs)
    a = 1

    # == [2] Run
    print("\n\n The recorder is recording the test data from the deployment. Press \'q\' to stop the recording.")
    data_collector.run()

    # == [3] End the recording and compressing
    terminate_process_and_children(rosbag_proc)

    # == [4] Stop the roscore if started from the script
    if roscore_proc is not None:
        terminate_process_and_children(roscore_proc)
   
