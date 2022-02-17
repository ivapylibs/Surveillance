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

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
import camera.utils.display as display
from camera.utils.writer import frameWriter

from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

from Surveillance.utils.utils import assert_directory
import Surveillance.layers.scene as scene
import Surveillance.layers.human_seg as Human_Seg
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg



@dataclass
class Params:
    markerLength: float = 0.01  # The aruco tag side length in meter
    save_dir: str = None  # the directory for data saving. Only for the data generated during the deployment
    calib_data_save_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cache_base"
    )
    reCalibrate: bool = True  # re-calibrate the system or use the previous data
    visualize = True        # Visualize the running process or not, including the source data and the processing results
    # NOTE: the two radius above can be upgraded to be adaptive

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
            scene_interpreter (scene.SceneInterpreterV1): The scene interpreter .
            params (Params, optional): The parameter passsed. Defaults to Params().
        """
        self.imgSource = imgSource
        self.scene_interpreter = scene_interpreter
        self.params = params

        # the saving directories
        self.save_dir = self.params.save_dir 
        assert_directory(directory=self.save_dir)

        # visualization option:
        self.visualize = self.params.visualize

        # storage for the processing result
        self.img_BEV = None
        self.humanImg = None
        self.puzzleImg = None

    def run(self):
        while (True):
            # ready = input("Please press \'r\' when you have placed the puzzles on the table")
            rgb, dep, status = self.imgSource()
            if not status:
                raise RuntimeError("Cannot get the image data")

            # measure 
            self.measure(rgb, dep)

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

    def vis(self, rgb, dep):
        # print("Visualize the scene")
        # self.scene_interpreter.vis_scene()
        # plt.show()

        # visualize the source data
        display.display_rgb_dep_cv(rgb, dep, ratio=0.4, window_name="Camera feed")

        # visualize the scene interpreter result
        display.display_images_cv([self.puzzleImg[:,:,::-1], self.humanImg[:,:,::-1]], ratio=0.4)

        self.vis_results(rgb, dep)
    
    def vis_results(self, rgb, dep):
        """Overwrite to put any result-related visualization in this function

        Args:
            rgb (_type_): _description_
            dep (_type_): _description_
        """
        return

    def save_data(self):
        """Overwrite to put any data saving code in this function
        """
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
        intrinsic = np.array(
            [[1.38106177e+03, 0.00000000e+00, 9.78223145e+02],
             [0.00000000e+00, 1.38116895e+03, 5.45521362e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        )

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

        # depth preprocess

        # run the calibration routine
        scene_interpreter = scene.SceneInterpreterV1.buildFromSource(
            lambda: d435_starter.get_frames()[:2],
            # d435_starter.intrinsic_mat,
            intrinsic,
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

        return BaseSurveillanceDeploy(d435_starter.get_frames, scene_interpreter, params)


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

