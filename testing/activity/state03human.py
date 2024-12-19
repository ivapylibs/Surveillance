#=================================== state03human ==================================
## @file
# @brief         Test the extraction of the human hand states from the Surveillance output
# 
# @ingroup  TestSurveillance_Layers
# 
# @author         Yiye Chen.          yychen2019@gatech.edu
# @date           2022/02/12
# 
# @todo     Test data loading is not working since the npz file is too big to
#           fit into the memory.  Need to reduce the file size first.  Now test
#           state parser on the simulation data first.
#
# @quitf
#
#=================================== state03human ==================================

from dataclasses import dataclass
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

import camera.d435.runner as d435
from camera.utils.display import display_rgb_dep_cv 
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
from camera.utils.writer import frameWriter

from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

import camera.utils.display as display
import Surveillance.layers.scene as scene 
import Surveillance.layers.human_seg as Human_Seg
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg

@dataclass
class Params:
    markerLength: float = 0.01       # The aruco tag side length in meter
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

class SceneInterpreterRunner():
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

        self.img_BEV = None
        self.humanImg = None
        self.puzzleImg = None
        self.meaBoardMask = None
        self.meaBoardImg = None

    def run(self):
        while(True):
            #ready = input("Please press \'r\' when you have placed the puzzles on the table")
            rgb, dep, status = self.imgSource()

            # measure 
            self.measure(rgb, dep)

            # visualize
            self.vis_results(rgb, dep)

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

        # get puzzles possibly interacted with the human
        self.get_near_hand_puzzles(rgb, dep, vis=True)

        # get the measure board
        meaBoardMask, meaBoardImg = self._get_measure_board()
        self.meaBoardMask = meaBoardMask
        self.meaBoardImg = meaBoardImg

        return meaBoardMask ,meaBoardImg
    
    def vis_results(self, rgb, dep):
        #print("Visualize the scene")
        #self.scene_interpreter.vis_scene()
        #plt.show()

        #TODO: could add a postprocess to filter the detected human hand scale
        if self.hTracker is not None:
            humanImg = cv2.circle(self.humanImg, 
                center=(int(self.hTracker[0]), int(self.hTracker[1])), 
                radius=20, 
                color=(0, 0, 255), 
                thickness=-1
            )
        else:
            humanImg = self.humanImg

        display.display_rgb_dep_cv(rgb, dep, ratio=0.4, window_name="Camera feed")
        display.display_images_cv([humanImg[:,:,::-1], self.puzzleImg[:,:,::-1]], ratio=0.4, \
            window_name="The human puzzle playing. Left: The human layer; Right: The puzzle layer")

    
    def save_data(self):
        self.frame_writer_meaImg.save_frame(self.meaBoardImg, None)
        # save the measure board img"
        self.frame_writer_meaMask.save_frame(self.meaBoardMask[:,:,np.newaxis].astype(np.uint8)*255, None)
        print("The data is saved")

    def get_near_hand_puzzles(self, rgb, dep, radius = None, vis=False):
        """Get the puzzle pieces that are possibly near to the hand
        
        It is done by drawing a circular region around the BEV_rectified hand centroid, 
        and then test whether the puzzle pieces locates within the region.

        The circular area is automatically determined by the distance between the fingure tip point and the centroid

        Args:
            rgb
            dep
            radius (int. Optional).     The radius of the near_hand region. Defaults to None, in which case will be auto-determined \
                by the furthest target hand color pixel from the centroid.
            vis (bool. Optional).  If set to True, then will visualize the hand + puzzle layer with the hand centroid, \
                circular region, and the high light of the possibly interacted puzzle pieces
        Returns:
            idx [np.ndarray].   The index of the puzzle pieces that is near to the hand. If none, then return None
        """
        hTracker_BEV = self.scene_interpreter.get_trackers("human", BEV_rectify=True)   # (2, 1)
        pTracker_BEV = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)  # (2, N)
        hand_mask = self.scene_interpreter.get_layer("human", mask_only=True, BEV_rectify=True)
        idx = np.empty((0))

        # if either hand or puzzle pieces are not presented, then return None
        if hTracker_BEV is None or pTracker_BEV is None or np.all(hand_mask == False):
            idx = None
        # otherwise
        else:
            # determine radius
            if radius is None:
                hy, hx = np.where(hand_mask)
                fingertip_idx = np.argmax(hy)
                r = ((hx[fingertip_idx] - hTracker_BEV[0])**2 + (hy[fingertip_idx] - hTracker_BEV[1])**2)**0.5
                #distances = np.sum(
                #    (np.concatenate((hx[np.newaxis, :], hy[np.newaxis, :]), axis=0) - hTracker_BEV)**2,
                #    axis=0
                #)
                #r = np.amin(distances) 
            else: r = radius
            #  get puzzle-to-human distances
            distances = np.sum(
                (pTracker_BEV - hTracker_BEV)**2,
                axis=0
            )**0.5
            near_hand = distances < r
            idx = np.where(near_hand)[0]
            #print(idx) 
        
        # visualization
        if vis:
            # The human + puzzle image
            img = self.scene_interpreter.get_layer("human", mask_only=False, BEV_rectify=True)[:,:,::-1] \
                + self.scene_interpreter.get_layer("puzzle", mask_only=False, BEV_rectify=True)[:,:,::-1]
            # if nothing is near hand
            if idx is not None:
                img = cv2.circle(img, hTracker_BEV.squeeze().astype(int), radius=int(r), color=(1, 0, 255), thickness=10)
                for i in idx:
                    img = cv2.circle(img, pTracker_BEV[:, i].squeeze().astype(int), radius=20, color=(0, 255, 0), thickness=-1)
            text = "Near-hand puzzles, which is the puzzle pieces locate within a circle around the hand."
            if radius is not None:
                text = text + " The radius is auto-computed"
            display.display_images_cv([img], ratio=0.5, window_name=text)
        return idx
    
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
        if x.size != 0:
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
            "cache_human_puzzle"
        )
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        cache_dir = os.path.join(
            fDir,
            "cache_human_puzzle/" + params.bg_color
        )

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
            varThreshold=30.,
            detectShadows=True,
            ShadowThreshold=0.6,
            postprocessor=lambda mask: mask
        )
        # parameters - robot
        robot_Params = Robot_Seg.Params()
        # parameters - puzzle
        kernel= np.ones((15,15), np.uint8)
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
            #d435_starter.intrinsic_mat,
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
            reCalibrate = params.reCalibrate,
            cache_dir=cache_dir
        )

        return SceneInterpreterRunner(d435_starter.get_frames, scene_interpreter, params)

def get_scene_interpreter(calib_data_path, configs):

    # camera intrinsic, fixed
    intrinsic = np.array(
        [[1.38106177e+03, 0.00000000e+00, 9.78223145e+02],
        [0.00000000e+00, 1.38116895e+03, 5.45521362e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    )

    # The BEV mat 
    BEV_mat_path = os.path.join(calib_data_path, "BEV_mat.npz")
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
        varThreshold=30.,
        detectShadows=True,
        ShadowThreshold=0.6,
        postprocessor=lambda mask: mask
    )

    # parameters - robot
    robot_Params = Robot_Seg.Params()

    # parameters - puzzle
    kernel= np.ones((15,15), np.uint8)
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
        None,
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
        reCalibrate = False,
        cache_dir=calib_data_path
    )

    return scene_interpreter

if __name__ == "__main__":

    # == Create a Scene Interpreter

    # data path - It contains both the calibration data and the test video
    fPath = os.path.dirname(os.path.abspath(__file__))
    tPath = os.path.dirname(fPath)
    dPath = os.path.join(tPath, "data/activity_1")

    # Configs
    configs = Params(
        markerLength = 0.08,
    )

    scene_interpreter = get_scene_interpreter(dPath, configs)


    # == Load the test video and depth
    vid_file_name = os.path.join(dPath, "puzzle_play.avi")
    dep_file_name = os.path.join(dPath, "puzzle_play.npz")
    cap = cv2.VideoCapture(vid_file_name)
    depths = np.load(dep_file_name, mmap_mode='r')["depth_frames"]   # use the mmap, so that a large file is stored in the disk rather than the memory
    frame_count = 0

    # == Create the state parser

    # == Run
    while(cap.isOpened()):
        # get the data
        ret, frame = cap.read()
        rgb = frame[:,:,::-1]
        dep = depths[frame_count, :, :]
        display_rgb_dep_cv(rgb, dep)

        # First get the centroid

        # then parse the state
   
#
#=================================== state03human ==================================
