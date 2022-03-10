"""

    @ brief:        The Base class for deploying the Surveillance system.
                    It defines the default parameters, encompass the calibration process,
                    and defines the API for further development.

                    The test script at the end will do nothing other than performing the Surveillance system calibration 
                    run on the incoming camera signals, and retrieve the processing result for visualization

    @author:        Yiye Chen,          yychen2019@gatech.edu
    @date:          02/16/2022

"""
from dataclasses import dataclass
import cv2
import numpy as np
import os
import time

# ROS related library
import rospy
import rosbag
from std_msgs.msg import Float64
from cv_bridge import CvBridge

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
import camera.utils.display as display

from ROSWrapper.publishers.Matrix_pub import Matrix_pub
from ROSWrapper.publishers.Image_pub import Image_pub
from ROSWrapper.subscribers.preprocess.matrix import multiArray_to_np

from Surveillance.utils.utils import assert_directory
import Surveillance.layers.scene as scene
from Surveillance.deployment.utils import depth_to_before_scale
from Surveillance.deployment.default_params import *



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
    ros_pub: bool = True         # Publish the test data to ros or not
    #### The calibration topics 
    # deployment - camera info
    BEV_mat_topic: str = "BEV_mat"
    intrinsic_topic: str = "intrinsic"
    depth_scale_topic: str = "depth_scale"
    # scene interpreter
    empty_table_rgb_topic: str = "empty_table_rgb"
    empty_table_dep_topic: str = "empty_table_dep"
    glove_rgb_topic: str = "glove_rgb"
    human_wave_rgb_topic: str = "human_wave_rgb"
    human_wave_dep_topic: str = "human_wave_dep"
    #### The test data topics
    test_rgb_topic: str = "test_rgb"
    test_depth_topic: str = "test_depth"
    #### Run the system on the test data or not
    run_system: bool = True
    # Will be stored in the class. Will be initiated in the building process
    depth_scale: float = None

class BaseSurveillanceDeploy():
    def __init__(self, imgSource, scene_interpreter: scene.SceneInterpreterV1, params: Params = Params()) -> None:
        """
        The Base class for deploying the Surveillance system.
        It defines the default parameters, encompasses the calibration process,
        and defines the API for further development.

        Args:
            imgSource (Callable): The image source that is able to get the camera data in the following style \
                (where status is a binary indicating whether the camera data is fetched successfully): 
                rgb, dep, status = imgSource()
                Can pass None, which will disable the run API that deploy the system on the connected camera
            scene_interpreter (scene.SceneInterpreterV1): The scene interpreter .
            params (Params, optional): The parameter passed. Defaults to Params().
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

        # control the rate
        self.rate = rospy.Rate(10) # hard code 10 FPS for now

    def run(self):

        if self.params.run_system is True:
            flag_process = True
        else:
            flag_process = False

        while (True):
            # ready = input("Please press \'r\' when you have placed the puzzles on the table")
            rgb, dep, status = self.imgSource()
            if not status:
                raise RuntimeError("Cannot get the image data")


            if flag_process == True:
                # process
                self.process(rgb, dep)

                # visualize
                if self.visualize:
                    # with results
                    self.vis(rgb, dep)
            else:
                # visualize
                if self.visualize:
                    self.vis_input(rgb, dep)

            # save data
            opKey = cv2.waitKey(1)
            if opKey == ord("c") and self.params.run_system is False:
                print("Recording process starts.")
                flag_process = True
            if opKey == ord("q"):
                print("Recording process ends.")
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

            # post process - NOTE: The postprocess for the puzzle solver is done here.
            self.postprocess(rgb, dep)

        # publish data - TODO: sometimes the program stuck here
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


    def publish_data(self):
        test_depth_bs = depth_to_before_scale(self.test_depth, self.depth_scale, np.uint16)
        self.test_rgb_pub.pub(self.test_rgb)
        self.test_dep_pub.pub(test_depth_bs)
        self.rate.sleep()
        

    def vis(self, rgb, dep):
        # print("Visualize the scene")
        # self.scene_interpreter.vis_scene()
        # plt.show()

        # visualize the source data
        display.display_rgb_dep_cv(rgb, dep, ratio=0.4, window_name="Camera feed")

        # visualize any results desired
        if self.params.run_system:
            self.vis_results(rgb, dep)

    def vis_input(self, rgb, dep):

        # visualize the source data
        display.display_rgb_dep_cv(rgb, dep, ratio=0.4, window_name="Camera feed")
    
    def vis_results(self, rgb, dep):
        """Overwrite to put any result-related visualization in this function

        Args:
            rgb (_type_): _description_
            dep (_type_): _description_
        """
        display.display_images_cv([self.humanImg, self.puzzleImg], ratio=0.4, window_name="Surveillance Process results")
        return

    def save_data(self):
        raise NotImplementedError

    def postprocess(self, rgb, dep):
        """Overwrite to put any pose process that is built on top of the scene interpreter her

        Args:
            rgb (_type_): _description_
            dep (_type_): _description_
        """

        # post process for the puzzle solver
        meaBoardMask, meaBoardImg = self._get_measure_board()
        self.meaBoardMask = meaBoardMask
        self.meaBoardImg = meaBoardImg

    def _get_measure_board(self, board_type = "test"):
        """
        Compare to the puzzle segmentation mask, the measure board carves out a larger circular area
        around each puzzle piece region to get high recall.
        """

        mea_test_r = 50
        mea_sol_r = 300

        puzzle_seg_mask = self.scene_interpreter.get_layer("puzzle", mask_only=True, BEV_rectify=True)
        puzzle_tpt = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)

        # initialize a blank mask for the measured board
        meaBoardMask = np.zeros_like(puzzle_seg_mask, dtype=np.uint8)

        if board_type=="test":
            # if some puzzle pieces are tracked
            if puzzle_tpt is not None:
                # get the centroid mask. Note the tpt is in opencv coordinate system
                centroids = puzzle_tpt.astype(int)
                cols = centroids[0, :]
                rows = centroids[1, :]
                rows[rows >= self.img_BEV.shape[0]] = self.img_BEV.shape[0] - 1
                cols[cols >= self.img_BEV.shape[1]] = self.img_BEV.shape[1] - 1

                # dilate-based method
                # meaBoardMask[rows, cols] = 1
                ## dilate with circle
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.params.mea_test_r,self.params.mea_test_r))
                # mask_proc = maskproc(
                #    maskproc.dilate, (kernel,)
                # )
                # meaBoardMask = mask_proc.apply(meaBoardMask)

                # circle-based method
                for i in range(rows.size):
                    meaBoardMask = cv2.circle(meaBoardMask, (cols[i], rows[i]), mea_test_r,
                                              color=(255, 255, 255), thickness=-1)

                # @note Yunzhi: Hack the region of visible area
                meaBoardMask[:400, :] = 0
                meaBoardMask[-250:, :] = 0
                meaBoardMask[:, :400] = 0
                meaBoardMask = (meaBoardMask != 0)
        else:
            # get the centroid
            x, y = np.where(puzzle_seg_mask)
            if x.size != 0:
                meaBoardMask[int(np.mean(x)), int(np.mean(y))] = 1

                ## dilate with circle
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.params.mea_sol_r, self.params.mea_sol_r))
                # mask_proc = maskproc(
                #    maskproc.dilate, (kernel,)
                # )
                # meaBoardMask = mask_proc.apply(meaBoardMask)

                # circle-based
                meaBoardMask = cv2.circle(meaBoardMask, (int(np.mean(y)), int(np.mean(x))),
                                          radius=mea_sol_r,
                                          color=(255, 255, 255), thickness=-1)

                meaBoardMask = (meaBoardMask != 0)

        meaBoardImg = meaBoardMask[:, :, np.newaxis].astype(np.uint8) * self.img_BEV

        # NOTE: remove the hand/robot mask that might be included due to the circular enlargement
        hand_mask = self.scene_interpreter.get_layer("human", mask_only=True, BEV_rectify=True)
        robot_mask = self.scene_interpreter.get_layer("robot", mask_only=True, BEV_rectify=True)
        meaBoardMask[hand_mask] = 0   # remove the hand mask
        meaBoardMask[robot_mask] = 0  # remove the robot mask

        meaBoardImg = meaBoardMask[:, :, np.newaxis].astype(np.uint8) * meaBoardImg

        return meaBoardMask, meaBoardImg

    @staticmethod
    def build(params: Params = Params()):
        """
        Builder for saving the calibration data in the local cache folder.

        Args:
            params (Params, optional):  The parameter passed. Defaults to Params().

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
                                                window_name ='Camera pose estimation',
                                                instruction="Camera pose estimation: \n Please place the Aruco tag close to the base for the Extrinsic and Bird-eye-view(BEV) matrix calibration. \n Press \'c\' to start the process. \n Please remove the tag upon completion.",
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
    def buildPub(params: Params = Params(), bag_path=None):
        """
        Builder for publishing the calibration data to ROS.

        Args:
            params (Params, optional): The deployment parameters. Defaults to Params().
                If params.reCalibrate is False, then will read the rosbag files for the calibration data to build the system,
                then run on the camera data
            bag_path (str): The rosbag file path. Necessary if the params.reCalibrate is False. Defaults to None

        Returns:
            _type_: _description_
        """

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
        depth_scale = d435_starter.get("depth_scale")

        # The aruco-based calibrator
        calibrator_CtoW = CtoW_Calibrator_aruco(
            d435_starter.intrinsic_mat,
            distCoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            markerLength_CL=params.markerLength,
            maxFrames=10,
            stabilize_version=True
        )

        # == [2] If do not wish to reCalibrate the system, read the rosbag and run
        if not params.reCalibrate:
            assert bag_path is not None
            runner = BaseSurveillanceDeploy.buildFromRosbag(bag_path=bag_path, params=params)
            runner.imgSource = d435_starter.get_frames     
            return runner
        
        # == [3] build a scene interpreter by running the calibration routine

        # prepare the publishers - TODO: add the M_CL and robot to world transformation. Probably to tf, which will still be able to record by rosbag
        BEV_mat_topic = params.BEV_mat_topic
        intrinsic_topic = params.intrinsic_topic
        depth_scale_topic = params.depth_scale_topic

        BEV_pub = Matrix_pub(BEV_mat_topic)   
        intrinsic_pub = Matrix_pub(intrinsic_topic)
        depth_scale_pub = rospy.Publisher(depth_scale_topic, Float64)
        time.sleep(5)   # to ensure the publisher is properly established

        # publish the intrinsic matrix and the depth scale
        intrinsic_pub.pub(d435_starter.intrinsic_mat)
        depth_scale_msg = Float64()
        depth_scale_msg.data = depth_scale
        depth_scale_pub.publish(depth_scale_msg)

        # Calibration begins
        print("\n\n")
        print("===========Calibration starts===========")
        print("\n")

        # calibrate the BEV transformation and publish
        rgb, dep = display.wait_for_confirm(lambda: d435_starter.get_frames()[:2],
                                            color_type="rgb",
                                            ratio=0.5,
                                            window_name='[1] Camera pose estimation',
                                            instruction="[1] Camera pose estimation: \n Please place the Aruco tag close to the base for the Extrinsic and Bird-eye-view(BEV) matrix calibration. \n Press \'c\' to start the process. \n Please remove the tag upon completion.",
                                            )
        while not calibrator_CtoW.stable_status:
            rgb, dep, _ = d435_starter.get_frames()
            M_CL, corners_aruco, img_with_ext, status = calibrator_CtoW.process(rgb, dep)
            assert status, "The aruco tag can not be detected"
        # calibrate the BEV_mat
        topDown_image, BEV_mat = BEV_rectify_aruco(rgb, corners_aruco, target_pos="down", target_size=100,
                                                    mode="full")
        BEV_pub.pub(BEV_mat)

        # sleep a while for the topic to be published

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
            nonROI_region=NONROI_FUN(params.H, params.W),
        )

        params.depth_scale = depth_scale
        params.ros_pub = True
        return BaseSurveillanceDeploy(d435_starter.get_frames, scene_interpreter, params)

    @staticmethod
    def buildFromRosbag(bag_path, params:Params):
        """Build the deployment runner instance

        Args:
            bag_path: The path of the Rosbag file
            params (Params, optional): The deployment parameters. Defaults to Params().

        Returns:
            _type_: _description_
        """

        bag = rosbag.Bag(bag_path)

        # prepare the publisher if necessary
        if params.ros_pub:
            # prepare the publishers - TODO: add the M_CL and robot to world transformation. Probably to tf, which will still be able to record by rosbag
            BEV_mat_topic = params.BEV_mat_topic
            intrinsic_topic = params.intrinsic_topic
            depth_scale_topic = params.depth_scale_topic

            BEV_pub = Matrix_pub(BEV_mat_topic)   
            intrinsic_pub = Matrix_pub(intrinsic_topic)
            depth_scale_pub = rospy.Publisher(depth_scale_topic, Float64)
            time.sleep(5)   # to ensure the publisher is properly established 
        
        # == [1] Load the scene information

        # get the BEV matrix and intrinsic matrix
        for topic, msg, t in bag.read_messages(["/"+params.BEV_mat_topic]):
            BEV_mat = multiArray_to_np(msg, (3, 3)) 
        intrinsic = None
        for topic, msg, t in bag.read_messages(["/"+params.intrinsic_topic]):
            intrinsic = multiArray_to_np(msg, (3, 3)) 
        if intrinsic is None:
            print("There is no intrinsic matrix stored in the bag. Will use the intrinsic of the D435 1920x1080")
            intrinsic = np.array(
                [[1.38106177e3, 0, 9.78223145e2],
                [0, 1.38116895e3, 5.45521362e2],
                [0., 0., 1.]]
            )

        # get the depth scale
        for topic, msg, t in bag.read_messages(["/"+params.depth_scale_topic]):
            depth_scale = msg.data
        
        # publish the BEV_matrix, intrinsic, and the depth scale
        if params.ros_pub:
            BEV_pub.pub(BEV_mat)
            intrinsic_pub.pub(intrinsic)
            depth_scale_msg = Float64()
            depth_scale_msg.data = depth_scale
            depth_scale_pub.publish(depth_scale)
            time.sleep(1)   # sleep to make sure they are published


        # == [3] Build the scene interpreter
        # get the camera size for nonROI_region determination
        cv_bridge = CvBridge()
        wait_time = 1
        print(params.empty_table_rgb_topic)
        for topic, msg, t in bag.read_messages(["/"+params.empty_table_rgb_topic]):
            empty_table_rgb = cv_bridge.imgmsg_to_cv2(msg)
            H, W = empty_table_rgb.shape[:2]

        scene_interpreter = scene.SceneInterpreterV1.buildFromRosbag(
            bag_path,
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
            ros_pub = params.ros_pub,
            empty_table_rgb_topic = params.empty_table_rgb_topic,
            empty_table_dep_topic = params.empty_table_dep_topic,
            glove_rgb_topic = params.glove_rgb_topic,
            human_wave_rgb_topic = params.human_wave_rgb_topic,
            human_wave_dep_topic = params.human_wave_dep_topic,
            depth_scale=depth_scale,
            intrinsic=intrinsic,
            nonROI_region=NONROI_FUN(H, W),
        )

        params.depth_scale = depth_scale
        return BaseSurveillanceDeploy(None, scene_interpreter, params)

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
