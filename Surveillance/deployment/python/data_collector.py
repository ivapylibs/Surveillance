"""

    @brief:     The data collector that saves all the calibration data and the test data.
                It will run through the calibration process for the Surveillance system, store the calibration data,
                and then save the test data(captured during the deployment) WITHOUT running the surveillance system on the test data

                The aim is to collect the data (corresponding calibration and test data) for the testing files that
                depends on the Surveillance system (e.g. activity analysis). The current surveillance system is not
                realtime, hence does not run it on the test data during the collection.

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           02/12/2022

"""

from dataclasses import dataclass
import cv2
import numpy as np
import os,sys

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
from camera.utils.writer import frameWriter, vidWriter
from camera.utils.writer_ros import vidWriter_ROS
import camera.utils.display as display

from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

import Surveillance.layers.scene as scene 
import Surveillance.layers.human_seg as Human_Seg
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg
from Surveillance.utils.utils import assert_directory
from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams

@dataclass
class Params(bParams):
    frame_rate: int = 30                         # The frame rate.
    depth_scale_topic: str = "depth_scale"       # The topic name for saving out the depth scale to the rosbag
    depth_topic: str = "depth"                   # The topic name for saving out the depth map to the rosbag
    rgb_topic: str = "color"                     # The topic name for saving out the rgb image to the rosbag
    depth_scale: float = None                   # Will be updated later
    rosbag_name: str = "data.bag"               # The rosbag file name

class DataCollector(BaseSurveillanceDeploy):
    """The data collector only collect the calibration data (that describes the scene) and the video during the deployment
    It won't run the system during the deployment. Instead, the only option is to save it out.

    The data will be saved as the rosbag.
    """
    def __init__(self, imgSource, scene_interpreter: scene.SceneInterpreterV1, params: Params = Params()) -> None:
        super().__init__(imgSource=imgSource, scene_interpreter=scene_interpreter, params=params)
        assert params.depth_scale is not None
        self.vid_writer = vidWriter_ROS(
            save_file_path=os.path.join(params.save_dir, params.rosbag_name),
            depth_scale=params.depth_scale,
            depth_scale_topic=params.depth_scale_topic,
            rgb_topic=params.rgb_topic,
            depth_topic=params.depth_topic,
            frame_rate=params.frame_rate
        )
        self.depth_scale = params.depth_scale

    def vis_results(self, rgb, dep):
        return

    def save_data(self, rgb, dep):
        self.vid_writer.save_frame(rgb, dep)
    
    def run(self):
        """Overwrite the runner API to only save the data but not run the system
        """
        while (True):
            # ready = input("Please press \'r\' when you have placed the puzzles on the table")
            rgb, dep, status = self.imgSource()
            if not status:
                raise RuntimeError("Cannot get the image data")

            # measure 
            #self.measure(rgb, dep)

            # visualize
            if self.visualize:
                self.vis(rgb, dep*self.depth_scale)

            # save data
            self.save_data(rgb, dep)

            opKey = cv2.waitKey(1)
            if opKey == ord("q"):
                self.finish()
                break

    def finish(self):
        self.vid_writer.finish()
    
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
            W_color=1920,
            H_color=1080,
            exposure=100,
            gain=55
        )

        d435_starter = d435.D435_Runner(d435_configs)

        # update the depth scale
        params.depth_scale = d435_starter.get("depth_scale")

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

        # parameters - human
        human_params = Human_Seg.Params(
            det_th=8,
            postprocessor=lambda mask: \
                cv2.dilate(
                    mask.astype(np.uint8),
                    np.ones((10, 10), dtype=np.uint8),
                    1
                ).astype(bool)
        )
        # parameters - tabletop
        bg_seg_params = Tabletop_Seg.Params_GMM(
            history=300,
            NMixtures=5,
            varThreshold=30.,
            detectShadows=True,
            ShadowThreshold=0.6,
            postprocessor=lambda mask: mask
        )
        # parameters - robot
        robot_Params = Robot_Seg.Params()
        # parameters - puzzle
        kernel = np.ones((15, 15), np.uint8)
        mask_proc_puzzle_seg = maskproc(
            maskproc.opening, (kernel,),
            maskproc.closing, (kernel,),
        )
        puzzle_params = Puzzle_Seg.Params_Residual(
            postprocessor=lambda mask: \
                mask_proc_puzzle_seg.apply(mask.astype(bool))
        )

        # trackers - human
        human_tracker = centroid.centroid(
            params=centroid.Params(
                plotStyle="bo"
            )
        )

        # trackers - puzzle
        puzzle_tracker = mCentroid.centroidMulti(
            params=mCentroid.Params(
                plotStyle="rx"
            )
        )

        # run the calibration routine
        scene_interpreter = scene.SceneInterpreterV1.buildFromSource(
            lambda: d435_starter.get_frames(before_scale=False)[:2],
            d435_starter.intrinsic_mat,
            #intrinsic,
            rTh_high=1,
            rTh_low=0.02,
            hTracker=human_tracker,
            pTracker=puzzle_tracker,
            hParams=human_params,
            rParams=robot_Params,
            pParams=puzzle_params,
            bgParams=bg_seg_params,
            params=scene.Params(
                BEV_trans_mat=BEV_mat
            ),
            reCalibrate=params.reCalibrate,
            cache_dir=cache_dir
        )

        return DataCollector(lambda: d435_starter.get_frames(before_scale=False), scene_interpreter, params)

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
    puzzle_data_collector = DataCollector.build(configs)
    

    # == [2] Deploy
    puzzle_data_collector.run()
   
