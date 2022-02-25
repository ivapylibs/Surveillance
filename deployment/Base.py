"""

    @ brief:        The Base class for deploying the Surveillance system.
                    It defines the default parameters, encompass the calibration process,
                    and defines the API for further developement.

                    The test script at the end will do nothing other than performing the Surveillance system calibration 
                    run on the incoming camera signals, and retrieve the processing result for visualization

    @author:        Yiye Chen,          yychen2019@gatech.edu
    @date:          02/16/2022

"""
from dataclasses import dataclass
import cv2
import numpy as np
import os
import rospy
from std_msgs.msg import Float64

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
import camera.utils.display as display

from ROSWrapper.publishers.Matrix_pub import Matrix_pub
from ROSWrapper.publishers.Image_pub import Image_pub

from Surveillance.utils.utils import assert_directory
import Surveillance.layers.scene as scene
from default_params import *

# util function
def depth_to_before_scale(depth, scale, dtype):
    depth_before_scale = depth / scale
    depth_before_scale = depth_before_scale.astype(dtype)
    return depth_before_scale

@dataclass
class Params:
    markerLength: float = 0.08  # The aruco tag side length in meter
    W: int = 1920               # The width of the frames
    H: int = 1080                # The depth of the frames
    save_dir: str = None        # the directory for data saving. Only for the data generated during the deployment
    calib_data_save_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cache_base"
    )
    reCalibrate: bool = True  # re-calibrate the system or use the previous data
    visualize: bool = True            # Visualize the running process or not, including the source data and the processing results
    depth_scale: float = None
    ros_pub: bool = True         # Publish the test data to the ros or not
    test_rgb_topic: str = "test_rgb"
    test_depth_topic: str = "test_depth"
    run_system: bool = True


class BaseSurveillanceDeploy():
    def __init__(self, imgSource, scene_interpreter: scene.SceneInterpreterV1, params: Params = Params()) -> None:
        """
        The Base class for deploying the Surveillance system.
        It defines the default parameters, encompass the calibration process,
        and defines the API for further developement.

        Args:
            imgSource (Callable): The image source that is able to get the camera data in the following style \
                (where status is a binary indicating whether the camera data is fetched successfully): 
                rgb, dep, status = imgSource()
                Can pass None if not required to run on the source
            scene_interpreter (scene.SceneInterpreterV1): The scene interpreter .
            params (Params, optional): The parameter passsed. Defaults to Params().
        """
        self.imgSource = imgSource
        self.scene_interpreter = scene_interpreter
        self.params = params

        # the saving directories
        self.save_dir = self.params.save_dir 

        # visualization option:
        self.visualize = self.params.visualize

        # storage for the processing result
        self.img_BEV = None
        self.humanImg = None
        self.puzzleImg = None

        # depth scale
        self.depth_scale = params.depth_scale

        # test data publishers
        self.test_rgb_pub = Image_pub(topic_name=params.test_rgb_topic)
        self.test_dep_pub = Image_pub(topic_name=params.test_depth_topic)

        # store the test data
        self.test_rgb = None
        self.test_depth = None

    def run(self):
        while (True):
            # ready = input("Please press \'r\' when you have placed the puzzles on the table")
            rgb, dep, status = self.imgSource()
            if not status:
                raise RuntimeError("Cannot get the image data")

            # process
            self.process(rgb, dep)

            # visualize
            if self.visualize:
                self.vis(rgb, dep)

            # save data
            opKey = cv2.waitKey(1)
            if opKey == ord("q"):
                break
            elif opKey == ord("s"):
                self.save_data()
            else:
                continue

    def process(self, rgb, dep):
        # save the data
        self.test_rgb = rgb
        self.test_depth = dep


        # measure the data
        if self.params.run_system:
            self.measure(rgb, dep)
        
        # visualize
        if self.visualize:
            self.vis(rgb, dep)

        # publish data
        if self.params.ros_pub:
            self.publish_data()


    def measure(self, rgb, dep):
        """
        get the measure board

        return the measure board mask and the measure board img
        """
        # interpret the scene
        self.scene_interpreter.process_depth(dep)
        self.scene_interpreter.process(rgb)
        self.img_BEV = cv2.warpPerspective(
            rgb.astype(np.uint8),
            self.scene_interpreter.params.BEV_trans_mat,
            (rgb.shape[1], rgb.shape[0])
        )

        # store important results
        self.puzzleImg = self.scene_interpreter.get_layer("puzzle", mask_only=False, BEV_rectify=True)
        self.humanImg = self.scene_interpreter.get_layer("human", mask_only=False, BEV_rectify=False)
        self.hTracker = self.scene_interpreter.get_trackers("human", BEV_rectify=False)

        # post process
        self.postprocess(rgb, dep)

    def publish_data(self):
        self.test_rgb_pub.pub(self.test_rgb)
        test_depth_bs = depth_to_before_scale(self.test_depth, self.depth_scale, np.uint16)
        self.test_dep_pub.pub(test_depth_bs)

    def vis(self, rgb, dep):
        # print("Visualize the scene")
        # self.scene_interpreter.vis_scene()
        # plt.show()

        # visualize the source data
        display.display_rgb_dep_cv(rgb, dep, ratio=0.4, window_name="Camera feed")

        # visualize any results desired
        self.vis_results(rgb, dep)
    
    def vis_results(self, rgb, dep):
        """Overwrite to put any result-related visualization in this function

        Args:
            rgb (_type_): _description_
            dep (_type_): _description_
        """
        return

    def save_data(self):
        raise NotImplementedError

    def postprocess(self, rgb, dep):
        """Overwrite to put any pose process that is built on top of the scene interpreter her

        Args:
            rgb (_type_): _description_
            dep (_type_): _description_
        """
        return

    @staticmethod
    def build(params: Params = Params()):
        # the cache folder for the data
        cache_dir = params.calib_data_save_dir
        if params.reCalibrate:
            assert_directory(cache_dir)
        
        # also assert the saving directory
        assert_directory(directory=params.save_dir)

        # camera runner
        d435_configs = d435.D435_Configs(
            W_dep=848,
            H_dep=480,
            W_color=params.W,
            H_color=params.H,
            exposure=100,
            gain=55
        )

        d435_starter = d435.D435_Runner(d435_configs)

        # The aruco-based calibrator
        calibrator_CtoW = CtoW_Calibrator_aruco(
            d435_starter.intrinsic_mat,
            distCoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            markerLength_CL=params.markerLength,
            maxFrames=10,
            stabilize_version=True
        )

        # == [2] build a scene interpreter by running the calibration routine

        # Calibrate the extrinsic matrix
        BEV_mat_path = os.path.join(cache_dir, "BEV_mat.npz")

        # Check if the calibration file is existing
        if not os.path.exists(BEV_mat_path):
            print('No calibration file exists. Start calibration.')
            params.reCalibrate = True

        if params.reCalibrate:

            rgb, dep = display.wait_for_confirm(lambda: d435_starter.get_frames()[:2],
                                                color_type="rgb",
                                                ratio=0.5,
                                                instruction="Camera pose estimation: \n Please place the Aruco tag close to the base for the Extrinsic and Bird-eye-view(BEV) matrix calibration. \n Press \'c\' to start the process. \n Please remove the tag upon completion",
                                                )
            while not calibrator_CtoW.stable_status:
                rgb, dep, _ = d435_starter.get_frames()
                M_CL, corners_aruco, img_with_ext, status = calibrator_CtoW.process(rgb, dep)
                assert status, "The aruco tag can not be detected"
            # calibrate the BEV_mat
            topDown_image, BEV_mat = BEV_rectify_aruco(rgb, corners_aruco, target_pos="down", target_size=100,
                                                       mode="full")
            # save
            np.savez(
                BEV_mat_path,
                BEV_mat=BEV_mat
            )
        else:
            print('Load the saved calibration files.')
            BEV_mat = np.load(BEV_mat_path, allow_pickle=True)["BEV_mat"]

        # run the calibration routine
        scene_interpreter = scene.SceneInterpreterV1.buildFromSource(
            lambda: d435_starter.get_frames()[:2],
            d435_starter.intrinsic_mat,
            #intrinsic,
            rTh_high=1,
            rTh_low=0.02,
            hTracker=HTRACKER,
            pTracker=PPARAMS,
            hParams=HPARAMS,
            rParams=ROBPARAMS,
            pParams=PPARAMS,
            bgParams=BGPARMAS,
            params=scene.Params(
                BEV_trans_mat=BEV_mat
            ),
            reCalibrate=params.reCalibrate,
            cache_dir=cache_dir
        )

        return BaseSurveillanceDeploy(d435_starter.get_frames, scene_interpreter, params)


    @staticmethod
    def buildPub(params: Params = Params(), ros_pub = True):
        """Build the deployment runner instance

        Args:
            params (Params, optional): The deployment parameters. Defaults to Params().
            ros_pub (bool, optional): If True, will publish the calibration data to a folder

        Returns:
            _type_: _description_
        """
        # the cache folder for the data
        cache_dir = params.calib_data_save_dir
        if params.reCalibrate:
            assert_directory(cache_dir)
        
        # also assert the saving directory
        assert_directory(directory=params.save_dir)

        # camera runner
        d435_configs = d435.D435_Configs(
            W_dep=848,
            H_dep=480,
            W_color=params.W,
            H_color=params.H,
            exposure=100,
            gain=55
        )

        d435_starter = d435.D435_Runner(d435_configs)

        # The aruco-based calibrator
        calibrator_CtoW = CtoW_Calibrator_aruco(
            d435_starter.intrinsic_mat,
            distCoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            markerLength_CL=params.markerLength,
            maxFrames=10,
            stabilize_version=True
        )
        
        # == [2] build a scene interpreter by running the calibration routine
        if params.reCalibrate:

            # prepare the publishers - TODO: add the M_CL and robot to world transformation. Probably to tf, which will still be able to record by rosbag
            BEV_mat_topic = "BEV_mat"
            intrinsic_topic = "intrinsic"
            BEV_pub = Matrix_pub(BEV_mat_topic)   
            intrinsic_pub = Matrix_pub(intrinsic_topic)

            # publish the intrinsic matrix and the depth scale
            intrinsic_pub.pub(d435_starter.intrinsic_mat)

            # calibrate the BEV transformation and publish
            rgb, dep = display.wait_for_confirm(lambda: d435_starter.get_frames()[:2],
                                                color_type="rgb",
                                                ratio=0.5,
                                                instruction="Camera pose estimation: \n Please place the Aruco tag close to the base for the Extrinsic and Bird-eye-view(BEV) matrix calibration. \n Press \'c\' to start the process. \n Please remove the tag upon completion",
                                                )
            while not calibrator_CtoW.stable_status:
                rgb, dep, _ = d435_starter.get_frames()
                M_CL, corners_aruco, img_with_ext, status = calibrator_CtoW.process(rgb, dep)
                assert status, "The aruco tag can not be detected"
            # calibrate the BEV_mat
            topDown_image, BEV_mat = BEV_rectify_aruco(rgb, corners_aruco, target_pos="down", target_size=100,
                                                       mode="full")
            BEV_pub.pub(BEV_mat)

            # run the calibration routine
            scene_interpreter = scene.SceneInterpreterV1.buildFromSourcePub(
                d435_starter,
                rTh_high=1,
                rTh_low=0.02,
                hTracker=HTRACKER,
                pTracker=PTRACKER,
                hParams=HPARAMS,
                rParams=ROBPARAMS,
                pParams=PPARAMS,
                bgParams=BGPARMAS,
                params=scene.Params(
                    BEV_trans_mat=BEV_mat
                ),
                ros_pub = True,
                empty_table_rgb_topic = "empty_table_rgb",
                empty_table_dep_topic = "empty_table_dep",
                glove_rgb_topic = "glove_rgb",
                human_wave_rgb_topic = "human_wave_rgb",
                human_wave_dep_topic = "human_wave_dep",
                depth_scale_topic = "depth_scale"
            )

            depth_scale = d435_starter.get("depth_scale")
            params.depth_scale = depth_scale
            params.ros_pub = True
            return BaseSurveillanceDeploy(d435_starter.get_frames, scene_interpreter, params)
        else:
            return BaseSurveillanceDeploy.buildFromRosbag()
            


    @staticmethod
    def buildFromRosbag(bag_path):
        raise NotImplementedError



if __name__ == "__main__":
    fDir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.dirname(fDir)

    # save_dir = os.path.join(save_dir, "data/puzzle_solver_black")
    save_dir = os.path.join(save_dir, "data/foo")

    # == [0] Configs
    configs = Params(
        markerLength=0.08,
        save_dir=save_dir,
        reCalibrate=False,
        calib_data_save_dir=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cache_base"
        )
    )

    # == [1] Prepare the camera runner & extrinsic calibrator
    baseSurveillance = BaseSurveillanceDeploy.build(configs)

    # == [2] Deploy
    baseSurveillance.run()

