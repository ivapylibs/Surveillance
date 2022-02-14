"""

    @brief:     The data collector that saves all the calibration data and the test data.
                It will run through the calibration process for the Surveillance system, store the calibration data,
                and then save the test data(captured during the deployment) WITHOUT run the surveillance system on the test data

                The aim is to collect the data (corresponding calibration and test data) for the testing files that depends on the Surveillance system 
                (e.g. activity analysis).
                The current surveillance system is not realtime, hence does not run it on the test data during the collection.

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           02/12/2022

"""

from dataclasses import dataclass
import cv2
import numpy as np
import os

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
from camera.utils.writer import frameWriter, vidWriter

from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

import Surveillance.utils.display as display
import Surveillance.layers.scene as scene 
import Surveillance.layers.human_seg as Human_Seg
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg

@dataclass
class Params:
    markerLength: float = 0.08       # The aruco tag side length in meter
    vidname:str = "SinglePiece"       # The test data name
    save_dir:str = None                     # the directory for data saving
    reCalibrate:bool = True             # re-calibrate the system or use the previous data
    W: int = 1920               # The width of the frames
    H: int = 1080                # The depth of the frames

class DataCollector():
    """The puzzle data collector built on top of the scene interpreter

    The collector will use the scene interpreter to segment the puzzle layer, 
    do some postprocess to get the data suitable for the puzzle solver,
    and then save the data
    """
    def __init__(self, imgSource, scene_interpreter:scene.SceneInterpreterV1, params:Params=Params()) -> None:
        self.imgSource = imgSource
        self.scene_interpreter = scene_interpreter
        self.params = params

        self.vidWriter = vidWriter(
            dirname=self.params.save_dir, 
            vidname=self.params.vidname, 
            W = self.params.W,
            H = self.params.H,
            save_depth=True # This is a must
        )

        self.start_save = False

    def run(self):
        while(True):
        
            #ready = input("Please press \'r\' when you have placed the puzzles on the table")
            rgb, dep, status = self.imgSource()
            self.img = rgb

            # measure - Do not measure! Just data collection
            #self.measure(rgb, dep)

            # visualize
            #print("Visualize the scene")
            #self.scene_interpreter.vis_scene()
            #plt.show()

            if self.start_save:
                text = "Saving out the frames... Press \'q\' to end"
            else:
                text ="Press \'s\' to start recording. Press \'q\' to end" 

            display.display_rgb_dep_cv(rgb, dep, ratio=0.4, window_name=text)

            # save frames
            self.vidWriter.save_frame(rgb, dep)

            # save data
            opKey = cv2.waitKey(1)
            if self.start_save:
                if opKey == ord("q"):
                    break 
                else:
                    continue
            else:
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
                    (self.scene_interpreter.BEV_size[1], self.scene_interpreter.BEV_size[0])
                )

        # get the measure board
        meaBoardMask, meaBoardImg = self._get_measure_board()
        self.meaBoardMask = meaBoardMask
        self.meaBoardImg = meaBoardImg
        return meaBoardMask ,meaBoardImg
    
    def save_data(self):
        # TODO: add the single frame data collection here.
        return
        print("The data is saved")
    
    @staticmethod
    def build(params:Params=Params()):
        # the cache folder for the data
        cache_dir = params.save_dir

        # camera runner
        d435_configs = d435.D435_Configs(
            W_dep=848,
            H_dep=480,
            W_color=params.W,
            H_color=params.H,
            exposure=100,
            gain=55 
        )

        # intrinsic
        intrinsic_mat_path = os.path.join(cache_dir, "intrinsic_mat.npz")
        if params.reCalibrate:
            d435_starter = d435.D435_Runner(d435_configs)
            intrinsic_mat =  d435_starter.intrinsic_mat
            np.savez(
                intrinsic = intrinsic_mat
            )
        else:
            intrinsic_mat = np.load(intrinsic_mat_path, allow_pickle=True)["intrinsic"]

        # The aruco-based calibrator
        calibrator_CtoW = CtoW_Calibrator_aruco(
            intrinsic_mat,
            distCoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            markerLength_CL = params.markerLength,
            maxFrames = 10,
            stabilize_version = True
        )

        # == [2] build a scene interpreter by running the calibration routine
        print("Calibrating the Surveillance system...")

        # calibrate the extrinsic matrix
        BEV_mat_path = os.path.join(cache_dir, "BEV_mat.npz")
        if params.reCalibrate:
            rgb, dep = display.wait_for_confirm(lambda: d435_starter.get_frames()[:2], 
                    color_type="rgb", 
                    ratio=0.5,
                    instruction="Please place the Aruco tag close to the base for the Extrinsic and Bird-eye-view(BEV) matrix calibration. Press \'c\' to confirm. Please remove the tag after the next calibration item starts.",
            )
            while not calibrator_CtoW.stable_status:
                rgb, dep, _ = d435_starter.get_frames()
                M_CL, corners_aruco, img_with_ext, status = calibrator_CtoW.process(rgb, dep)
                assert status, "The aruco tag can not be detected"
            # calibrate the BEV_mat
            topDown_image, BEV_mat = BEV_rectify_aruco(rgb, corners_aruco, target_pos="down", target_size=100, mode="full")
            # save
            np.savez(
                BEV_mat_path,
                BEV_mat = BEV_mat 
            )
        else:
            BEV_mat = np.load(BEV_mat_path, allow_pickle=True)["BEV_mat"]

        # parameters - human
        human_params = Human_Seg.Params(
            det_th=8,
            postprocessor= lambda mask:\
                cv2.dilate(
                    mask.astype(np.uint8),
                    np.ones((10,10), dtype=np.uint8),
                    1
                ).astype(bool)
        )
        # parameters - tabletop
        bg_seg_params = Tabletop_Seg.Params_GMM(
            history=300,
            NMixtures=5,
            varThreshold=100.,
            detectShadows=True,
            ShadowThreshold=0.55,
            postprocessor=lambda mask: mask
        )
        # parameters - robot
        robot_Params = Robot_Seg.Params()
        # parameters - puzzle
        kernel= np.ones((9,9), np.uint8)
        mask_proc_puzzle_seg = maskproc(
            maskproc.opening, (kernel, ),
            maskproc.closing, (kernel, ),
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
        puzzle_tracker=mCentroid.centroidMulti(
                params=mCentroid.Params(
                    plotStyle="rx"
                )
            )

        # run the calibration routine
        scene_interpreter = scene.SceneInterpreterV1.buildFromSource(
            lambda: d435_starter.get_frames()[:2],
            intrinsic_mat,
            rTh_high=1,
            rTh_low=0.02,
            hTracker=human_tracker,
            pTracker=puzzle_tracker,
            hParams=human_params,
            rParams=robot_Params,
            pParams=puzzle_params,
            bgParams=bg_seg_params,
            params=scene.Params(
                BEV_trans_mat=BEV_mat,
                BEV_rect_size=topDown_image.shape[:2]
            ),
            reCalibrate = params.reCalibrate,
            cache_dir=cache_dir
        )

        return DataCollector(d435_starter.get_frames, scene_interpreter, params)

if __name__ == "__main__":

    fDir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.dirname(
        os.path.dirname(fDir)
    )
    # save_dir = os.path.join(save_dir, "data/puzzle_solver_black")
    save_dir = os.path.join(save_dir, "testing/data/activity_2")
    # == [0] Configs
    configs = Params(
        markerLength = 0.08,
        save_dir = save_dir,
        vidname = "puzzle_play",    
        reCalibrate = True,          
        W = 1920,
        H = 1080
    )


    # == [1] Prepare the camera runner & extrinsic calibrator
    puzzle_data_collector = DataCollector.build(configs)
    

    # == [2] Deploy
    puzzle_data_collector.run()
   
