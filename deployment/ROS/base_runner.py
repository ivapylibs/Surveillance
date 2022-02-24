"""

    @brief          The Base Runner that build the Surveillance system and run on the Color&Depth topic.

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           02/23/2022

"""

from dataclasses import dataclass
import cv2
import numpy as np
import os
import sys
import rospy
from std_msgs.msg import Float32

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
from camera.utils.writer import frameWriter
import camera.utils.display as display 

from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

from ROSWrapper.subscribers.Images_sub import Images_sub


import Surveillance.layers.scene as scene



deployPath = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(deployPath)
from Base import BaseSurveillanceDeploy
from Base import Params as bParams


@dataclass
class Params(bParams):
    rgb_topic: str = "color"
    depth_topic: str = "depth"
    depth_scale_topic: str = "depth_scale"


class SurvROSBase(BaseSurveillanceDeploy):
    def __init__(self, imgSource, scene_interpreter: scene.SceneInterpreterV1, params: Params = Params()) -> None:
        super().__init__(imgSource=imgSource, scene_interpreter=scene_interpreter, params=params)

        self.depth_scale = None

        self.imgs_sub = Images_sub([params.rgb_topic, params.depth_topic], callback_np=self.run_args)
        self.depth_scale_sub = rospy.Subscriber(params.depth_scale_topic, Float32)

    def measure_cb(self, img_list):
        rgb = img_list[0]
        dep = img_list[1]
        if dep.dtype == np.float:
            dep = dep
        elif dep.dtype == np.int:
            dep = dep.astype(np.float32) * self.depth_scale

        self.measure(rgb, dep)
        if self.visualize:
            self.vis(rgb, dep)

            opKey = cv2.waitKey(1)
            if opKey == ord("q"):
                exit()

    
    def set_depth_scale(self, depth_scale_msg:Float32):
        self.depth_scale = depth_scale_msg.data
   


if __name__ == "__main__":
    fDir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.dirname(
        os.path.dirname(fDir)
    )
    # save_dir = os.path.join(save_dir, "data/puzzle_solver_black")
    save_dir = os.path.join(save_dir, "data/data_collect")

    # == [0] Configs
    configs = Params(
        markerLength = 0.08,
        calib_data_save_dir = save_dir,
        save_dir = save_dir,
        reCalibrate = False,          
        W = 1920,
        H = 1080,
        frame_rate = 30,                
        depth_scale_topic = "depth_scale",
        depth_topic = "depth",
        rgb_topic = "color",
        depth_scale = None
    )


    # == [1] Prepare the camera runner & extrinsic calibrator
    base_runner = SurvROSBase.build(configs)

