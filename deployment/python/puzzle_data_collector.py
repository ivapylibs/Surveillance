"""

    @brief          The puzzle solver data collector

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           10/08/2021

"""

from dataclasses import dataclass
from posixpath import dirname
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
from camera.utils.writer import frameWriter

from improcessor.mask import mask as maskproc
from numpy.core.shape_base import block
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
    save_name:str = "SinglePiece"       # when saving out, what to name to use
    save_dir:str = None                     # the directory for data saving
    bg_color:str = "white"              # white or black, depending on whether the black mat is applied
    reCalibrate:bool = True             # re-calibrate the system or use the previous data
    board_type:str = "test"          # test board or the solution board
    mea_test_r: int = 150               # The raidus of the puzzle carving on the test board (i.e. needs to carve out single puzzles)
                                        # If set to None, will return the segmentation board directly
    mea_sol_r: int = 300                # The raidus of the puzzle carving on the test board (i.e. needs to carve out multiple puzzle pieces)
                                        # If set to None, will return the segmentation board directly
    # NOTE: the two radius above can be upgraded to be adaptive

class PuzzleDataCollector():
    """The puzzle data collector built on top of the scene interpreter

    The collector will use the scene interpreter to segment the puzzle layer, 
    do some postprocess to get the data suitable for the puzzle solver,
    and then save the data
    """
    def __init__(self, imgSource, scene_interpreter:scene.SceneInterpreterV1, params:Params=Params()) -> None:
        self.imgSource = imgSource
        self.scene_interpreter = scene_interpreter
        self.params = params

        self.frame_writer_orig = frameWriter(
            dirname=self.params.save_dir, 
            frame_name=self.params.save_name+"_original", 
            path_idx_mode=True
        )
        self.frame_writer_seg = frameWriter(
            dirname=self.params.save_dir, 
            frame_name=self.params.save_name+"_seg",
            path_idx_mode=True
        )
        self.frame_writer_meaImg = frameWriter(
            dirname=self.params.save_dir, 
            frame_name=self.params.save_name+"_mea",
            path_idx_mode=True
        )
        self.frame_writer_meaMask = frameWriter(
            dirname=self.params.save_dir, 
            frame_name=self.params.save_name+"_meaMask",
            path_idx_mode=True
        )

        self.img = None
        self.img_BEV = None
        self.segImg = None
        self.meaBoardMask = None
        self.meaBoardImg = None

    def run(self):
        while(True):
        
            #ready = input("Please press \'r\' when you have placed the puzzles on the table")
            rgb, dep, status = self.imgSource()
            self.img = rgb

            # measure 
            self.measure(rgb, dep)

            # visualize
            #print("Visualize the scene")
            #self.scene_interpreter.vis_scene()
            #plt.show()

            self.segImg = self.scene_interpreter.get_layer("puzzle", mask_only=False, BEV_rectify=True)
            display.display_rgb_dep_cv(rgb, dep, ratio=0.5, window_name="Camera feed")
            display.display_images_cv([rgb[:,:,::-1], self.segImg[:,:,::-1], self.meaBoardImg[:,:,::-1]], ratio=0.3, \
                window_name="The puzzle layer")

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

        # get the measure board
        meaBoardMask, meaBoardImg = self._get_measure_board()
        self.meaBoardMask = meaBoardMask
        self.meaBoardImg = meaBoardImg
        return meaBoardMask ,meaBoardImg
    
    def save_data(self):
        # save the original data
        self.frame_writer_orig.save_frame(self.img, None)
        # save the segmentation image
        self.frame_writer_seg.save_frame(self.segImg, None)
        # save the measure board mask
        self.frame_writer_meaImg.save_frame(self.meaBoardImg, None)
        # save the measure board img"
        self.frame_writer_meaMask.save_frame(self.meaBoardMask[:,:,np.newaxis].astype(np.uint8)*255, None)
        print("The data is saved")
    
    def _get_measure_board(self):
        """
        Compare to the puzzle segmentation mask, the measure board carves out a larger circular area
        around each puzzle piece region to get high recall.
        """
        if self.params.board_type == "test":
            meaBoardMask, meaBoardImg = self._get_measure_board_test()
        elif self.params.board_type == "solution":
            meaBoardMask, meaBoardImg = self._get_measure_board_sol()
        return meaBoardMask, meaBoardImg

    def _get_measure_board_test(self):
        puzzle_seg_mask = self.scene_interpreter.get_layer("puzzle", mask_only=True, BEV_rectify=True)
        puzzle_tpt = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)

        # initialize the measure board mask  
        meaBoardMask = np.zeros_like(puzzle_seg_mask, dtype=bool)
        # if some puzzle piece are tracked
        if puzzle_tpt is not None:
            # get the centroid mask. Note the tpt is in opencv coordinate system
            centroids = puzzle_tpt.astype(int)
            cols = centroids[0,:]
            rows = centroids[1, :]
            rows[rows >= self.img_BEV.shape[0]] = self.img_BEV.shape[0] - 1
            cols[cols >= self.img_BEV.shape[1]] = self.img_BEV.shape[1] - 1
            meaBoardMask[rows, cols] = 1

            # dilate with circle
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.params.mea_test_r,self.params.mea_test_r))
            mask_proc = maskproc(
                maskproc.dilate, (kernel,)
            )
            meaBoardMask = mask_proc.apply(meaBoardMask)

        # finally obtain the meaBoardImg
        meaBoardImg = meaBoardMask[:,:,np.newaxis].astype(np.uint8)*self.img_BEV
        return  meaBoardMask, meaBoardImg

    def _get_measure_board_sol(self):
        # get the puzzle segmentation mask, trackpointers, and the img_BEV
        puzzle_seg_mask = self.scene_interpreter.get_layer("puzzle", mask_only=True, BEV_rectify=True)
        puzzle_tpt = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)

        # initialize the measure board mask  
        meaBoardMask = np.zeros_like(puzzle_seg_mask, dtype=bool)
        # get the centroid
        x,y = np.where(puzzle_seg_mask)
        meaBoardMask[int(np.mean(x)), int(np.mean(y))] = 1
        # dilate with circle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(self.params.mea_sol_r, self.params.mea_sol_r))
        mask_proc = maskproc(
            maskproc.dilate, (kernel,)
        )
        meaBoardMask = mask_proc.apply(meaBoardMask)
        # finally obtain the meaBoardImg
        meaBoardImg = meaBoardMask[:,:,np.newaxis].astype(np.uint8)*self.img_BEV
        return  meaBoardMask, meaBoardImg

    @staticmethod
    def build(params:Params=Params()):
        # the cache folder for the data
        fDir = os.path.dirname(
            os.path.realpath(__file__)
        )
        cache_dir = os.path.join(
            fDir,
            "cache_puzzle_data_collect/" + params.bg_color
        )

        # camera runner
        d435_configs = d435.D435_Configs(
            W_dep=1280,
            H_dep=720,
            W_color=1920,
            H_color=1080,
            exposure=100,
            gain=55 
        )

        d435_starter = d435.D435_Runner(d435_configs)

        # The aruco-based calibrator
        calibrator_CtoW = CtoW_Calibrator_aruco(
            d435_starter.intrinsic_mat,
            distCoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            markerLength_CL = 0.067,
            maxFrames = 10,
            stabilize_version = True
        )

        # == [2] build a scene interpreter by running the calibration routine
        print("Calibrating the scene interpreter, please wait...")

        # calibrate the extrinsic matrix
        rgb, dep, status = d435_starter.get_frames()
        M_CL, corners_aruco, img_with_ext = calibrator_CtoW.process(rgb, dep)
        topDown_image, BEV_mat = BEV_rectify_aruco(rgb, corners_aruco, returnMode=1) 

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
            varThreshold=15.,
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
            d435_starter.intrinsic_mat,
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
            reCalibrate = params.reCalibrate,
            cache_dir=cache_dir
        )

        return PuzzleDataCollector(d435_starter.get_frames, scene_interpreter, params)

if __name__ == "__main__":

    fDir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.dirname(
        os.path.dirname(fDir)
    )
    # save_dir = os.path.join(save_dir, "data/puzzle_solver_black")
    save_dir = os.path.join(save_dir, "data/yunzhi_test2")
    # == [0] Configs
    configs = Params(
        save_dir = save_dir,
        save_name = "ExplodedWithRotationAndExchange",    
        bg_color = "black",   # black or white        
        reCalibrate = True,          
        board_type = "test",    # test or solution         
        mea_test_r = 125,             
        mea_sol_r = 200               
    )


    # == [1] Prepare the camera runner & extrinsic calibrator
    puzzle_data_collector = PuzzleDataCollector.build(configs)
    

    # == [2] Deploy
    puzzle_data_collector.run()
   