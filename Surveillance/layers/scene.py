"""
 ============================== scene ===============================

    @brief              The scene interpreter for the puzzle playing task
                    
    The scene interpreter will split the scene into three four layers:
        1. Background (tabletop) layer
        2. Human layer
        3. Robot arm layer
        4. Puzzle piece layer
    The first three relys on their own segmenter, and the puzzle piece layer
    is assumed to be the residual.

    The interpreter will provide the following additional functions:
    1. Bird-eye-view rectification
                    
    
    @author:    Yiye Chen       yychen2019@gatech.edu
    @date:      09/16/2021

 ============================== scene ===============================
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, List
import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# processor
from improcessor.mask import mask as maskproc

# detectors
from Surveillance.utils.height_estimate import HeightEstimator
import camera.utils.display as display
import Surveillance.layers.human_seg as hSeg
import Surveillance.layers.robot_seg as rSeg 
import Surveillance.layers.tabletop_seg as tSeg
import Surveillance.layers.puzzle_seg as pSeg 

# camera utility
from camera.base import Base
from camera.utils.writer import frameWriter,  vidWriter
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco

# ROSWrapper
from ROSWrapper.publishers.Image_pub import Image_pub
import rospy
from std_msgs.msg import Float64
import rosbag
from cv_bridge import CvBridge, CvBridgeError

def depth_to_before_scale(depth, scale, dtype):
    depth_before_scale = depth / scale
    depth_before_scale = depth_before_scale.astype(dtype)
    return depth_before_scale

@dataclass
class Params():
    """
    Should be the parameters different from the ones used in the layer segmenters

    Args:
        BEV_trans_mat            The Bird-eye-view transformation matrix
        BEV_rect_size            np.ndarray. (2, ) The image size (H, W) after rectification. If None, then will use the input image size. Defaults to None

    """
    BEV_trans_mat: np.ndarray = None 
    BEV_rect_size: np.ndarray = None
    depth_preprocess: Callable = lambda dep: dep

class SceneInterpreterV1():
    """
    The scene interpreter will split the scene into three four layers:
        1. Background (tabletop) layer
        2. Human layer
        3. Robot arm layer
        4. Puzzle piece layer
    The first three relys on their own segmenter, and the puzzle piece layer
    is assumed to be the residual.

    The interpreter will provide the following additional functions:
    1. Bird-eye-view rectification

    @param[in]  human_seg           The human segmenter.
    @param[in]  robot_seg           The robot segmenter.
    @param[in]  bg_seg              The background segmenter.
    @param[in]  params              Other parameters
    @param[in]  nonROI_init         A mask of initial nonROI region, which will be always treated as the background
    """
    def __init__(self, 
                human_seg: hSeg.Human_ColorSG_HeightInRange, 
                robot_seg: rSeg.robot_inRange_Height, 
                bg_seg: tSeg.tabletop_GMM, 
                puzzle_seg: pSeg.Puzzle_Residual,
                heightEstimator: HeightEstimator, 
                params: Params,
                nonROI_init: np.ndarray = None
                ):
        self.params = params

        self.height_estimator = heightEstimator

        # segmenters
        self.human_seg = human_seg
        self.robot_seg = robot_seg
        self.bg_seg = bg_seg
        self.puzzle_seg = puzzle_seg 

        # cached processing info
        self.rgb_img = None          #<- The image that is lastly processed
        self.depth = None            #<- The depth map that is lastly processed
        self.height_map = None       #<- The height map that is lastly processed

        # the masks to store
        self.nonROI_init = nonROI_init      #<- The prior nonROI region
        self.nonROI_mask = None             #<- The nonROI for each frame, which is the prior + pixels without necessary information. (e.g. The depth is missing)
        self.bg_mask = None
        self.human_mask = None
        self.robot_mask = None
        self.puzzle_mask = None

        # the tracker state
        self.human_track_state = None
        self.puzzle_track_state = None
        self.robot_track_state = None

        # the BEV image size. If None, will be updated to be the input image size during application
        self.BEV_size = self.params.BEV_rect_size

    def process_depth(self, depth):
        """
        Process the depth map
        """
        self.depth = depth 
        self.height_map = np.abs(self.height_estimator.apply(depth))
        # update the height_map to those that requires
        self.human_seg.update_height_map(self.height_map)
        self.robot_seg.update_height_map(self.height_map)
    
    def process(self, img):
        """
        @param[in]  img         The rbg image to be processed
        """
        # For now might only need to implement this one. 
        # Just let the detectors to process the image one-by-one

        self.rgb_img = img

        # update the BEV size
        if self.BEV_size is None:
            self.BEV_size = img.shape[:2]

        # non-ROI
        self.nonROI_mask = self.get_nonROI()

        # human
        self.human_seg.process(img)
        self.human_mask = self.human_seg.get_mask()
        self.human_mask[self.nonROI_mask] = False
        self.human_track_state = self.human_seg.get_state()
        # bg
        self.bg_seg.process(img)
        self.bg_mask = self.bg_seg.get_mask()
        self.bg_mask = self.bg_mask | self.nonROI_mask
        self.bg_mask = self.bg_mask & (~self.human_mask)    #<- Trust human mask more
        # robot
        self.robot_seg.process(img)
        self.robot_mask = self.robot_seg.get_mask()
        self.robot_mask[self.human_mask] = False            #<- Trust the human mask and the bg mask more
        self.robot_mask[self.bg_mask] = False
        self.robot_track_state = self.robot_seg.get_state()
        # puzzle
        self.puzzle_seg.set_detected_masks([self.bg_mask, self.human_mask, self.robot_mask])
        self.puzzle_seg.process(img)
        self.puzzle_mask = self.puzzle_seg.get_mask()
        self.puzzle_track_state = self.puzzle_seg.get_state()
    
    def measure(self, img):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
    
    def correct(self):
        raise NotImplementedError
    
    def adapt(self):
        raise NotImplementedError
    
    def get_nonROI(self):
        """
        This function encode the prior knowledge of which region is not of interest

        Current non-ROI: 
        1. Depth is zero, which indicate failure in depth capturing
        2. Height is too big (above 0.5 meter), which indicates non-table region

        @param[out] mask        The binary mask indicating non-ROI
        """
        assert (self.height_map is not None) and (self.depth is not None)

        if self.nonROI_init is not None:
            mask = copy.deepcopy(self.nonROI_init)
        else:
            mask = np.zeros_like(self.depth, dtype=bool)
        # non-ROI 1 - The region with no depth
        mask[np.abs(self.depth) < 1e-3] = 1
        # non-ROI 2
        mask[self.height_map > 0.5] = 1

        return mask
    
    def get_trackers(self, layer_name, BEV_rectify=False):
        """Get the track pointers for a layer. 
        If no tracker is applied or no trackpointers are detected, then will return None

        Args:
            layer_name (str): The name of the layer trackers to get. \
                    Choices = ["human", "robot", "puzzle"]
            BEV_rectify (bool, optional): Rectify to the bird-eye-view or not. Defaults to False.

        Returns:
            tpt [np.ndarray, (2, N)]: The tracker pointers of the layer
        """
        # get the layer tracker state. Background is not permitted
        assert layer_name in ["human", "robot", "puzzle"]
        track_state = eval("self."+layer_name+"_track_state")

        # if the tracker is applied
        if track_state is not None:
            # if have no measurement, set return to none
            if not track_state.haveMeas:
                tpt = None
            # if have measurement,
            else:
                tpt = track_state.tpt
                # BEV rectify
                if BEV_rectify:
                    # API Requires the shape (1, N, D). See:https://stackoverflow.com/questions/45817325/opencv-python-cv2-perspectivetransform
                    tpt = cv2.perspectiveTransform(
                        tpt.T[np.newaxis, :, :],
                        self.params.BEV_trans_mat
                    )[0].T
        # if the tracker is not applied
        else:
            tpt = None

        return tpt


    def get_layer(self, layer_name, mask_only=False, BEV_rectify=False):
        """
        Get the content or the binary mask of a layer

        @param[in]  layer_name          The name of the layer mask to get
                                        Choices = ["bg", "human", "robot", "puzzle"]
        @param[in]  mask_only           Binary. If true, will get the binary mask
        @param[in]  BEV_rectify         Binary. If true, will rectify the layer
                                        to the bird-eye-view before return
        """
        # choices
        assert layer_name in ["bg", "human", "robot", "puzzle"]

        mask = eval("self."+layer_name+"_mask")

        if mask_only:
            layer = mask
        else:
            layer = mask[:,:, np.newaxis].astype(np.uint8) * self.rgb_img

        if BEV_rectify:
            assert self.params.BEV_trans_mat is not None, \
                "Please store the Bird-eye-view transformation matrix into the params"
            if self.params.BEV_rect_size is None:
                xsize = layer.shape[1]
                ysize = layer.shape[0]
            else:
                xsize = self.params.BEV_rect_size[1]
                ysize = self.params.BEV_rect_size[0]
            layer = cv2.warpPerspective(
                layer.astype(np.uint8), 
                self.params.BEV_trans_mat,
                (xsize, ysize)
            )
        
        if mask_only:
            layer = layer.astype(bool)
        
        return layer
    
    def vis_layer(self, layer_name, mask_only:bool=False, BEV_rectify:bool=False, 
                ax:plt.Axes=None):
        """
        Visualize the layer

        @param[in]  layer_name          The name of the layer mask to visualize
                                        Choices = ["bg", "human", "robot", "puzzle"]
        @param[in]  mask_only           Binary. If true, will visualize the binary mask
        @param[in]  BEV_rectify         Binary. If true, will rectify the layer
                                        to the bird-eye-view for visualization 
        @param[in]  ax                  The axis for visualization
        """
        # choices
        assert layer_name in ["bg", "human", "robot", "puzzle"]

        # ax
        if ax is None:
            plt.figure()
            ax = plt.gca()
        
        # set the title
        title = layer_name
        if BEV_rectify:
            title = title + "_BEV(Bird-eye-view)"
        ax.set_title(title)

        # display directly if needs no BEV 
        if not BEV_rectify:
            seg = eval("self."+layer_name+"_seg")
            seg.draw_layer(self.rgb_img, ax=ax)
        # if needs the BEV, then need to rectify the layer and the track pointers first
        else:
            # the layer
            layer = self.get_layer(layer_name, mask_only=mask_only, BEV_rectify=BEV_rectify)
            ax.imshow(layer)
            # the trackpointer
            seg = eval("self."+layer_name+"_seg")   #<- still needs it for visualization parameters
            tpt = self.get_trackers(layer_name, BEV_rectify=BEV_rectify)
            if tpt is not None:
                state_vis = seg.tracker.getState()
                state_vis.tpt = tpt
                seg.tracker.displayState(state_vis, ax)

    def vis_scene(self, 
                mask_only:List[bool]=[False, False, False, False], 
                BEV_rectify:List[bool]=[False, False, False, True],
                fh = None
    ):
        """
        Visualize four layers ["bg", "human", "robot", "puzzle"]

        @param[in]  mask_only       A list of bool corresponding to the 4 layers above.
                                    If true, will only visualize the binary mask
        @param[in]  BEV_rectify     A list of bool corresponding to the 4 layers above.
                                    If true, will visualize the bird-eye-view of the layer
        @param[in]  fh              The figure handle. matplotlib Figure type
        """

        # prepare the plot
        if fh is None:
            fh = plt.figure(figsize=(20,4))
        
        ax1 = fh.add_subplot(151)
        ax2 = fh.add_subplot(152)
        ax3 = fh.add_subplot(153)
        ax4 = fh.add_subplot(154)
        ax5 = fh.add_subplot(155)
        axes = [ax2, ax3, ax4, ax5]

        # visualize the test image
        ax1.imshow(self.rgb_img)
        ax1.set_title("The test image")

        # visualize the four layers
        for idx, layer_name in enumerate(["human", "robot", "bg", "puzzle"]):
            self.vis_layer(
                layer_name, 
                mask_only[idx],
                BEV_rectify[idx],
                ax=axes[idx]
            )

        plt.tight_layout()

    def vis_puzzles(self, mask_only=False, BEV_rectify=True, fh=None):
        """Visualize the puzzle pieces segmentation result and the region carved around
        the puzzle pieces centroid. The latter may be used to pass to the puzzle solver.
        
        NOTE: move the function to another place? A better structure?

        Args:
            mask_only (bool, optional): only visualize the mask or not. Defaults to False.
            BEV_rectify (bool, optional): Rectify to the bird-eye-view or not. Defaults to True.
            fh ([type], optional): Figure handle. Defaults to None.
        """
        if fh is None:
            fh = plt.figure(figsize=(15,5))
        fh.suptitle("Puzzle data")
        ax1 = fh.add_subplot(131)
        ax2 = fh.add_subplot(132)
        ax3 = fh.add_subplot(133)

        # visualize the test image
        ax1.imshow(self.rgb_img)
        ax1.set_title("The test image")

        # visualize the raw puzzle segmentation result
        img_BEV = cv2.warpPerspective(
                self.rgb_img.astype(np.uint8), 
                self.params.BEV_trans_mat,
                (self.rgb_img.shape[1], self.rgb_img.shape[0])
            )
        puzzle_seg_mask = self.get_layer("puzzle", mask_only=True, BEV_rectify=True)
        ax2.imshow(puzzle_seg_mask[:,:,np.newaxis].astype(np.uint8)*img_BEV)
        ax2.set_title("The segmentation result")
        
        # the dilated result
        import improcessor.mask as maskproc
        # rectify the centroid
        state = self.puzzle_seg.tracker.getState()
        state.tpt = cv2.perspectiveTransform(
            state.tpt.T[np.newaxis, :, :],
            self.params.BEV_trans_mat
        ).squeeze().T

        # get the centroid mask
        centroids = state.tpt.astype(int)
        puzzle_solver_mask = np.zeros_like(puzzle_seg_mask, dtype=bool)
        puzzle_solver_mask[centroids[1,:], centroids[0,:]] = 1

        # dilate with circle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
        mask_proc = maskproc.mask(
            maskproc.mask.dilate, (kernel,)
        )
        puzzle_solver_mask = mask_proc.apply(puzzle_solver_mask)
        ax3.imshow(puzzle_solver_mask[:,:,np.newaxis].astype(np.uint8)*img_BEV)
        ax3.set_title("The puzzle measured board sent to the puzzle solver")

    @staticmethod
    def buildFromSourceDir(
        imgSource:Callable,
        intrinsic,
        rTh_high = 1.0,
        rTh_low = 0.02,
        hTracker = None,
        pTracker = None,
        rTracker = None,
        hParams: hSeg.Params  = hSeg.Params(),
        rParams: rSeg.Params = rSeg.Params(),
        pParams: pSeg.Params_Residual = pSeg.Params_Residual(),
        bgParams: tSeg.Params_GMM = tSeg.Params_GMM(),
        params: Params() = Params(),
        reCalibrate: bool = True,
        cache_dir: str = None
    ):
        """The interface for building the sceneInterpreterV1.0 from an image source.
        Given an image source which can provide the stream of the rgb and depth data,
        this builder will build the scene interpreter in the following process:

        1. Ask for an empty tabletop rgb and depth data.
        2. Use the depth to build a height estimator
        3. Ask for a target color glove rgb image
        4. Use the  target color glove rgb image and the tabletop rgb image to build the \
            human segmenter
        5. Build a tabletop segmenter
        6. Ask for the human to wave across the working area with the glove. \
            The rgb data will be used to calibrate the tabletop segmenter
        7. Build the robot segmenter and the puzzle 

        This function will save the calibration data to various files in the cache_dir, 
        and load from that directory when reCalibrate is set to False

        Args:
            imgSource (Callable): A callable for getting the rgb and depth image. \
                Could be the camera runner interface or the ROS subscriber
            intrinsic (np.ndarray. Shape:(3,3)): The camera intrinsic matrix
            rTh_high (float, optional): The upper height threshold for the robot segmenter. Defaults to 1.0.
            rTh_low (float, optional): The lower hieght threshold for the robot segmenter. Defaults to 0.02.
            hTracker ([type], optional): human tracker. Defaults to None.
            pTracker ([type], optional): puzzle tracker. Defaults to None.
            rTracker ([type], optional): robot tracker. Defaults to None.
            hParams (hSeg.Params, optional): human segmenter parameters. Defaults to hSeg.Params().
            rParams (rSeg.Params, optional): robot segmenter parameters. Defaults to rSeg.Params().
            pParams (pSeg.Params_Residual, optional): puzzle segmenter parameters. Defaults to pSeg.Params_Residual().
            bgParams (tSeg.Params_GMM, optional): background segmenter parameters. Defaults to tSeg.Params_GMM().
            params (Params, optional): the scene interpreter parameters. Defaults to Params().
            reCalibrate (bool, optional): Defaults to True. If set to True, will ignore previous calibration results and re-calibrate
            cache_dir (Srting, optional): the directory storing the calibration data. Defaults to None, in which case will need \
                manual calibration. Otherwise will directly look for the calibration data. If no desired data found, then will \
                still need manual calibration, where the data will be saved in the cache folder.
            publish_calib_data
        """

        # ==[0] prepare
        # the save paths
        save_mode = False
        calibrate = True 

        if cache_dir is None:
            calibrate = True
            save_mode = False
        else:
            save_mode = True
            empty_table_name = "empty_table"
            glove_name = "color_glove"
            hand_wave_vid_name = "hand_wave"

            empty_table_rgb_path = os.path.join(
                cache_dir,
                empty_table_name + ".png"
            )
            empty_table_dep_path = os.path.join(
                cache_dir,
                empty_table_name + ".npz"
            )
            glove_rgb_path = os.path.join(
                cache_dir,
                glove_name + ".png"
            )
            glove_dep_path = os.path.join(
                cache_dir,
                glove_name + ".npz"
            )
            hand_wave_vid_path = os.path.join(
                cache_dir,
                hand_wave_vid_name + ".avi"
            )
            if (not reCalibrate) and (os.path.exists(empty_table_rgb_path)):
                calibrate=False
            else: 
                calibrate = True

        # if calibrate, then prepare the writer
        if calibrate:
            rgb, dep = imgSource()
            frame_saver = frameWriter(
                dirname=cache_dir, 
                frame_name = None,
                path_idx_mode=False
            )

            vid_writer = vidWriter(
                dirname=cache_dir, 
                vidname=hand_wave_vid_name,
                W=rgb.shape[1],
                H=rgb.shape[0],
                activate=True,
                save_depth=False
            )

        
        # ==[1] get the empty tabletop rgb and depth data
        if calibrate:
            empty_table_rgb, empty_table_dep = display.wait_for_confirm(imgSource, color_type="rgb", 
                ratio=0.5,
                window_name="[2] Empty table modeling",
                instruction="[2] Empty table modeling: \n Please clear the workspace and take an image of the empty table. \n Press \'c\' to confirm.",
            )
            cv2.destroyAllWindows()

            if save_mode:
                calibrate = True
                frame_saver.frame_name = empty_table_name
                frame_saver.save_frame(empty_table_rgb, empty_table_dep)
        else:
            empty_table_rgb = cv2.imread(empty_table_rgb_path)[:,:,::-1]
            empty_table_dep = np.load(empty_table_dep_path, allow_pickle=True)["depth_frame"]
        

        # ==[2] Build the height estimator
        height_estimator = HeightEstimator(intrinsic=intrinsic)
        height_estimator.calibrate(empty_table_dep)

        # ==[3] Get the glove image
        #if (not save_mode) or (not os.path.exists(glove_rgb_path)):
        if calibrate:
            glove_rgb, glove_dep = display.wait_for_confirm(imgSource, color_type="rgb", 
                ratio=0.5,
                window_name="[3] Static colored glove modeling",
                instruction="[3] Static colored glove modeling: \n Please place the colored glove on the table. \n Press \'c\' key to confirm.",
            )
            cv2.destroyAllWindows()

            if save_mode:
                frame_saver.frame_name = glove_name 
                frame_saver.save_frame(glove_rgb, glove_dep)
        else:
            glove_rgb = cv2.imread(glove_rgb_path)[:,:,::-1]
            glove_dep = np.load(glove_dep_path, allow_pickle=True)["depth_frame"]

        # ==[4] Build the human segmenter
        human_seg = hSeg.Human_ColorSG_HeightInRange.buildImgDiff(
            empty_table_rgb, glove_rgb, 
            tracker=hTracker, params=hParams
        )
        
        # == [5] Build a GMM tabletop segmenter
        bg_seg = tSeg.tabletop_GMM.build(bgParams, bgParams) 

        # == [6] Calibrate 
        # prepare 
        #if (not save_mode) or not os.path.exists(hand_wave_vid_path):
        if calibrate:

            ready = False
            complete = False

            instruction = "[4] Dynamic colored glove modeling: \n Please wear the glove and wave over the working area. \n Press \'c\' to start the calibration and then Press \'c\' again to finish the calibration. Or press \'q\' to quit the program."
            print(instruction)

            # display
            while ((ready is not True) or (complete is not True)):
                rgb, dep = imgSource()
                display.display_rgb_dep_cv(rgb, dep, window_name='[4] Dynamic colored glove modeling', ratio=0.5)
                opKey = cv2.waitKey(1)

                # press key?
                if opKey == ord('c'):
                    # if ready is False, then change ready to True
                    if not ready:
                        ready = True
                        instruction = "Now the calibration has started. Press \'c\' to end the calibration"
                        cv2.destroyAllWindows()

                    # if already ready but not complete, then change complete to True
                    elif not complete:
                        complete = True
                        cv2.destroyAllWindows()

                # if ready, then calibrate the bg segmenter following the procedure
                if ready:
                    height_map = height_estimator.apply(dep)
                    human_seg.update_height_map(height_map)
                    human_seg.process(rgb)
                    fgMask = human_seg.get_mask()
                    BG_mask = ~fgMask

                    # process with the GT BG mask
                    rgb_train = np.where(
                        np.repeat(BG_mask[:,:,np.newaxis], 3, axis=2),
                        rgb, 
                        empty_table_rgb
                    )

                    # calibrate
                    bg_seg.calibrate(rgb_train)

                    if save_mode:
                        vid_writer.save_frame(rgb_train, dep)
            if save_mode:
                vid_writer.finish()
        else:
            bg_hand = cv2.VideoCapture(hand_wave_vid_path)
            ret = True
            while(bg_hand.isOpened() and ret):
                ret, rgb_train = bg_hand.read()
                if ret:
                    rgb_train = rgb_train[:,:,::-1]
                    bg_seg.calibrate(rgb_train)


        # == [7] robot detector and the puzzle detector
        robot_seg = rSeg.robot_inRange_Height(
            low_th=rTh_low,
            high_th=rTh_high,
            tracker = rTracker,
            params=rParams
        )

        puzzle_seg = pSeg.Puzzle_Residual(
            theTracker=pTracker,
            params=pParams
        )

        # == [9] The nonROI region
        empty_table_height = height_estimator.apply(empty_table_dep)
        ROI_mask1 = (empty_table_height < 0.05)
        kernel_refine = np.ones((20, 20), dtype=bool)
        kernel_erode = np.ones((100, 100), dtype=bool)
        mask_proc1 = maskproc(
            maskproc.opening, (kernel_refine, ),
            maskproc.closing, (kernel_refine, )
        )
        mask_proc2 = maskproc(
            maskproc.erode, (kernel_erode, )
        )
        ROI_mask2 = mask_proc1.apply(ROI_mask1)
        ROI_mask3 = mask_proc2.apply(ROI_mask2)

        # display
        #f, axes =  plt.subplots(1, 4, figsize=(20, 5))
        #axes[1].imshow(empty_table_height)
        #axes[1].set_title("The height map")
        #axes[0].imshow(empty_table_rgb)
        #axes[0].set_title("The empty table rgb image")
        #axes[2].imshow(ROI_mask2, cmap="gray")
        #axes[2].set_title("The zero height region as ROI")
        #axes[3].imshow(ROI_mask3, cmap="gray")
        #axes[3].set_title("The shrinked ROI. (Eroded)")
        #plt.tight_layout()
        #plt.show()
        #exit()


        # == [8] create the scene interpreter and return
        scene_interpreter = SceneInterpreterV1(
            human_seg,
            robot_seg,
            bg_seg,
            puzzle_seg,
            height_estimator,
            params,
            nonROI_init=~ROI_mask3
            #nonROI_init = None
        )

        return scene_interpreter
    
    @staticmethod
    def buildFromSourcePub(
        cam_runner: Base,
        rTh_high = 1.0,
        rTh_low = 0.02,
        hTracker = None,
        pTracker = None,
        rTracker = None,
        hParams: hSeg.Params  = hSeg.Params(),
        rParams: rSeg.Params = rSeg.Params(),
        pParams: pSeg.Params_Residual = pSeg.Params_Residual(),
        bgParams: tSeg.Params_GMM = tSeg.Params_GMM(),
        params: Params() = Params(),
        ros_pub: bool = True,
        empty_table_rgb_topic: str = "empty_table_rgb",
        empty_table_dep_topic: str = "empty_table_dep",
        glove_rgb_topic: str = "glove_rgb",
        human_wave_rgb_topic: str = "human_wave_rgb",
        human_wave_dep_topic: str = "human_wave_dep",
        nonROI_region = None
    ):
        """The interface for building the sceneInterpreterV1.0 from an image source.
        Given an image source which can provide the stream of the rgb and depth data,
        this builder will build the scene interpreter in the following process:

        1. Ask for an empty tabletop rgb and depth data.
        2. Use the depth to build a height estimator
        3. Ask for a target color glove rgb image
        4. Use the  target color glove rgb image and the tabletop rgb image to build the \
            human segmenter
        5. Build a tabletop segmenter
        6. Ask for the human to wave across the working area with the glove. \
            The rgb data will be used to calibrate the tabletop segmenter
        7. Build the robot segmenter and the puzzle 

        The builder provides the option to publish all the calibration data (and the depth scale) to ros opics

        Args:
            cam_runner: The camera runner
            intrinsic (np.ndarray. Shape:(3,3)): The camera intrinsic matrix
            rTh_high (float, optional): The upper height threshold for the robot segmenter. Defaults to 1.0.
            rTh_low (float, optional): The lower hieght threshold for the robot segmenter. Defaults to 0.02.
            hTracker ([type], optional): human tracker. Defaults to None.
            pTracker ([type], optional): puzzle tracker. Defaults to None.
            rTracker ([type], optional): robot tracker. Defaults to None.
            hParams (hSeg.Params, optional): human segmenter parameters. Defaults to hSeg.Params().
            rParams (rSeg.Params, optional): robot segmenter parameters. Defaults to rSeg.Params().
            pParams (pSeg.Params_Residual, optional): puzzle segmenter parameters. Defaults to pSeg.Params_Residual().
            bgParams (tSeg.Params_GMM, optional): background segmenter parameters. Defaults to tSeg.Params_GMM().
            params (Params, optional): the scene interpreter parameters. Defaults to Params().
            ros_pub (bool, optional):   If true, will publish the data to the ros. Defaults to True
        """

        # ==[0] prepare the publishers
        if ros_pub:
            empty_table_rgb_pub = Image_pub(empty_table_rgb_topic)
            empty_table_dep_pub = Image_pub(empty_table_dep_topic)
            glove_rgb_pub = Image_pub(glove_rgb_topic)
            human_wave_rgb_pub = Image_pub(human_wave_rgb_topic)
            human_wave_dep_pub = Image_pub(human_wave_dep_topic)
        
        # the depth scale and before_scale dtype
        depth_scale = cam_runner.get("depth_scale")
        _, dep, _ = cam_runner.get_frames(before_scale=True)
        depth_dtype = dep.dtype

        # imgSource and intrinsic
        imgSource = lambda: cam_runner.get_frames()[:2]
        intrinsic = cam_runner.intrinsic_mat

        # ==[1] get the empty tabletop rgb and depth data
        empty_table_rgb, empty_table_dep = display.wait_for_confirm(
            imgSource, 
            color_type="rgb", 
            ratio=0.5,
            window_name='[2] Empty table modeling',
            instruction="[2] Empty table modeling: \n Please clear the workspace and take an image of the empty table. \n Press \'c\' to confirm.",
        )
        cv2.destroyAllWindows()
        if ros_pub:
            empty_table_dep_bs = depth_to_before_scale(empty_table_dep, depth_scale, depth_dtype)
            empty_table_rgb_pub.pub(empty_table_rgb)
            empty_table_dep_pub.pub(empty_table_dep_bs)

        # ==[2] Build the height estimator
        height_estimator = HeightEstimator(intrinsic=intrinsic)
        height_estimator.calibrate(empty_table_dep)

        # ==[3] Get the glove image
        #if (not save_mode) or (not os.path.exists(glove_rgb_path)):
        glove_rgb, glove_dep = display.wait_for_confirm(
            imgSource,
            color_type="rgb", 
            ratio=0.5,
            window_name="[3] Static colored glove modeling",
            instruction="[3] Static colored glove modeling: \n Please place the colored glove on the table. \n Press \'c\' key to confirm.",
        )
        cv2.destroyAllWindows()
        if ros_pub:
            glove_rgb_pub.pub(glove_rgb)

        # ==[4] Build the human segmenter
        human_seg = hSeg.Human_ColorSG_HeightInRange.buildImgDiff(
            empty_table_rgb, glove_rgb, 
            tracker=hTracker, params=hParams
        )
        
        # == [5] Build a GMM tabletop segmenter
        bg_seg = tSeg.tabletop_GMM.build(bgParams, bgParams) 

        # == [6] Calibrate 
        # prepare 
        ready = False
        complete = False

        instruction = "[4] Dynamic colored glove modeling: \n Please wear the glove and wave over the working area. \n Press \'c\' to start the calibration and then Press \'c\' again to finish the calibration. Or press \'q\' to quit the program."
        print(instruction)

        # display
        while ((ready is not True) or (complete is not True)):
            rgb, dep = imgSource()
            display.display_rgb_dep_cv(rgb, dep, window_name='[4] Dynamic colored glove modeling', ratio=0.5)
            opKey = cv2.waitKey(1)

            # press key?
            if opKey == ord('c'):
                # if ready is False, then change ready to True
                if not ready:
                    ready = True
                    instruction = "Now the calibration has started. Press \'c\' to end the calibration"
                    print(instruction)
                    cv2.destroyAllWindows()

                # if already ready but not complete, then change complete to True
                elif not complete:
                    complete = True
                    cv2.destroyAllWindows()

            # if ready, then calibrate the bg segmenter following the procedure
            if ready:
                height_map = height_estimator.apply(dep)
                human_seg.update_height_map(height_map)
                human_seg.process(rgb)
                fgMask = human_seg.get_mask()
                BG_mask = ~fgMask

                # process with the GT BG mask
                rgb_train = np.where(
                    np.repeat(BG_mask[:,:,np.newaxis], 3, axis=2),
                    rgb, 
                    empty_table_rgb
                )

                # calibrate
                bg_seg.calibrate(rgb_train)

                if ros_pub:
                    dep_bs = depth_to_before_scale(dep, depth_scale, depth_dtype)
                    human_wave_rgb_pub.pub(rgb)
                    human_wave_dep_pub.pub(dep_bs)


        # == [7] robot detector and the puzzle detector
        robot_seg = rSeg.robot_inRange_Height(
            low_th=rTh_low,
            high_th=rTh_high,
            tracker = rTracker,
            params=rParams
        )

        puzzle_seg = pSeg.Puzzle_Residual(
            theTracker=pTracker,
            params=pParams
        )

        # == [9] The nonROI region - The non-table region
        if nonROI_region is None:
            empty_table_height = height_estimator.apply(empty_table_dep)
            ROI_mask1 = (empty_table_height < 0.05)
            kernel_refine = np.ones((20, 20), dtype=bool)
            kernel_erode = np.ones((100, 100), dtype=bool)
            mask_proc1 = maskproc(
                maskproc.opening, (kernel_refine, ),
                maskproc.closing, (kernel_refine, )
            )
            mask_proc2 = maskproc(
                maskproc.erode, (kernel_erode, )
            )
            ROI_mask2 = mask_proc1.apply(ROI_mask1)
            ROI_mask3 = mask_proc2.apply(ROI_mask2)
            nonROI_init = ~ROI_mask3
        else:
            nonROI_init = nonROI_region
        

        # == [8] create the scene interpreter and return
        scene_interpreter = SceneInterpreterV1(
            human_seg,
            robot_seg,
            bg_seg,
            puzzle_seg,
            height_estimator,
            params,
            nonROI_init=nonROI_init
        )

        return scene_interpreter
    
    @staticmethod
    def buildFromRosbag(
        rosbag_file,
        rTh_high = 1.0,
        rTh_low = 0.02,
        hTracker = None,
        pTracker = None,
        rTracker = None,
        hParams: hSeg.Params  = hSeg.Params(),
        rParams: rSeg.Params = rSeg.Params(),
        pParams: pSeg.Params_Residual = pSeg.Params_Residual(),
        bgParams: tSeg.Params_GMM = tSeg.Params_GMM(),
        params: Params() = Params(),
        reCalibrate: bool = True,
        cache_dir: str = None,
        ros_pub: bool = False,
        empty_table_rgb_topic: str = "empty_table_rgb",
        empty_table_dep_topic: str = "empty_table_dep",
        glove_rgb_topic: str = "glove_rgb",
        human_wave_rgb_topic: str = "human_wave_rgb",
        human_wave_dep_topic: str = "human_wave_dep",
        # some additional information required for the camera
        depth_scale: float = None,
        intrinsic = None,
        nonROI_region = None
    ):
        
        bag = rosbag.Bag(rosbag_file)
        cv_bridge = CvBridge()
        wait_time = 1

        # ==[0] prepare the publishers
        if ros_pub:
            empty_table_rgb_pub = Image_pub(empty_table_rgb_topic)
            empty_table_dep_pub = Image_pub(empty_table_dep_topic)
            glove_rgb_pub = Image_pub(glove_rgb_topic)
            human_wave_rgb_pub = Image_pub(human_wave_rgb_topic)
            human_wave_dep_pub = Image_pub(human_wave_dep_topic)

        # ==[1] get the empty tabletop rgb and depth data
        for topic, msg, t in bag.read_messages(["/"+empty_table_rgb_topic]):
            empty_table_rgb = cv_bridge.imgmsg_to_cv2(msg)
        for topic, msg, t in bag.read_messages(["/"+empty_table_dep_topic]):
            empty_table_dep = cv_bridge.imgmsg_to_cv2(msg) * depth_scale
        display.display_rgb_dep_cv(empty_table_rgb, empty_table_dep, ratio=0.4, window_name="The empty table data from the rosbag")
        cv2.waitKey(wait_time)

        # publish if necessary
        if ros_pub:
            empty_table_dep_bs = depth_to_before_scale(empty_table_dep, depth_scale, np.uint16) # hardcode the dtype for now
            empty_table_rgb_pub.pub(empty_table_rgb)
            empty_table_dep_pub.pub(empty_table_dep_bs)

        # ==[2] Build the height estimator
        height_estimator = HeightEstimator(intrinsic=intrinsic)
        height_estimator.calibrate(empty_table_dep)

        # ==[3] Get the glove image
        #if (not save_mode) or (not os.path.exists(glove_rgb_path)):
        for topic, msg, t in bag.read_messages(["/"+glove_rgb_topic]):
            glove_rgb = cv_bridge.imgmsg_to_cv2(msg)
        display.display_images_cv([glove_rgb[:,:,::-1]], ratio=0.4, window_name="The glove color data from the rosbag")
        cv2.waitKey(wait_time)

        if ros_pub:
            glove_rgb_pub.pub(glove_rgb)

        # ==[4] Build the human segmenter
        human_seg = hSeg.Human_ColorSG_HeightInRange.buildImgDiff(
            empty_table_rgb, glove_rgb, 
            tracker=hTracker, params=hParams
        )
        
        # == [5] Build a GMM tabletop segmenter
        bg_seg = tSeg.tabletop_GMM.build(bgParams, bgParams) 

        # == [6] Calibrate 
        rgb = None
        depth = None
        for topic, msg, t in bag.read_messages(topics=["/"+human_wave_rgb_topic, "/"+human_wave_dep_topic]):
            if topic == "/"+human_wave_rgb_topic:
                rgb = cv_bridge.imgmsg_to_cv2(msg)
            elif topic == "/"+human_wave_dep_topic:
                depth = cv_bridge.imgmsg_to_cv2(msg) * depth_scale

            # display if gathered both data
            if rgb is not None and depth is not None:
                # display data
                display.display_rgb_dep_cv(rgb, depth, depth_clip=0.08, ratio=0.4, window_name="The background calibration data")
                cv2.waitKey(1)

                # prepare for calib
                height_map = height_estimator.apply(depth)
                human_seg.update_height_map(height_map)
                human_seg.process(rgb)
                fgMask = human_seg.get_mask()
                BG_mask = ~fgMask
                # process with the GT BG mask
                rgb_train = np.where(
                    np.repeat(BG_mask[:,:,np.newaxis], 3, axis=2),
                    rgb, 
                    empty_table_rgb
                )
                # calibrate
                bg_seg.calibrate(rgb_train)

                # publish
                if ros_pub:
                    dep_bs = depth_to_before_scale(depth, depth_scale, np.uint8)    # hardcode the dtype for now
                    human_wave_rgb_pub.pub(rgb)
                    human_wave_dep_pub.pub(dep_bs)

                # reset
                rgb = None
                depth = None

        # == [7] robot detector and the puzzle detector
        robot_seg = rSeg.robot_inRange_Height(
            low_th=rTh_low,
            high_th=rTh_high,
            tracker = rTracker,
            params=rParams
        )

        puzzle_seg = pSeg.Puzzle_Residual(
            theTracker=pTracker,
            params=pParams
        )

        # == [9] The nonROI region - The non-table region
        if nonROI_region is None:
            empty_table_height = height_estimator.apply(empty_table_dep)
            ROI_mask1 = (empty_table_height < 0.05)
            kernel_refine = np.ones((20, 20), dtype=bool)
            kernel_erode = np.ones((100, 100), dtype=bool)
            mask_proc1 = maskproc(
                maskproc.opening, (kernel_refine, ),
                maskproc.closing, (kernel_refine, )
            )
            mask_proc2 = maskproc(
                maskproc.erode, (kernel_erode, )
            )
            ROI_mask2 = mask_proc1.apply(ROI_mask1)
            ROI_mask3 = mask_proc2.apply(ROI_mask2)
            nonROI_init = ~ROI_mask3
        else:
            nonROI_init = nonROI_region
        

        # == [8] create the scene interpreter and return
        scene_interpreter = SceneInterpreterV1(
            human_seg,
            robot_seg,
            bg_seg,
            puzzle_seg,
            height_estimator,
            params,
            nonROI_init=nonROI_init
        )

        cv2.destroyAllWindows()

        return scene_interpreter