"""

    @brief          The test script that run the system on a rosbag file,
                    include building the model and test the model

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           02/25/2022

"""

import numpy as np
import os
import sys
import rospy
from std_msgs.msg import Float32

from ROSWrapper.subscribers.Images_sub import Images_sub


import Surveillance.layers.scene as scene
from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams 
from Surveillance.deployment.ROS.utils import terminate_process_and_children

test_rgb_topic = "test_color"
test_dep_topic = "test_depth"

if __name__ == "__main__":
    fDir = os.path.dirname(os.path.realpath(__file__))
    rosbag_file = os.path.join(fDir, "data_2022-02-24-22-41-21.bag")

    # == [0] build from the rosbag
    configs = bParams(
        markerLength = 0.08,
        reCalibrate = False,          
        W = 1920,
        H = 1080,
        frame_rate = 30,                
        depth_scale_topic = "depth_scale",
        depth_topic = "depth",
        rgb_topic = "color",
        depth_scale = None
    )
    surv = BaseSurveillanceDeploy.buildFromRosbag(rosbag_file)

    # == [1] Prepare the camera runner 
    surv.pr

