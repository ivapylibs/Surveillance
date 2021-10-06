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
from typing import List
import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2

# detectors
from Surveillance.utils.height_estimate import HeightEstimator
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.robot_seg import robot_inRange_Height
from Surveillance.layers.tabletop_seg import tabletop_GMM
from Surveillance.layers.puzzle_seg import Puzzle_Residual


@dataclass
class Params():
    """
    Should be the parameters different from the ones used in the layer segmenters

    @param BEV_trans_mat            The Bird-eye-view transformation matrix
    """
    BEV_trans_mat: np.ndarray = None 

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
    """
    def __init__(self, 
                human_seg: Human_ColorSG_HeightInRange, 
                robot_seg: robot_inRange_Height, 
                bg_seg: tabletop_GMM, 
                puzzle_seg: Puzzle_Residual,
                heightEstimator: HeightEstimator, 
                params: Params):
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
        self.bg_mask = None
        self.human_mask = None
        self.robot_mask = None
        self.puzzle_mask = None

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

        # non-ROI
        nonROI_mask = self.nonROI()

        # human
        self.human_seg.process(img)
        self.human_mask = self.human_seg.get_mask()
        # bg
        self.bg_seg.process(img)
        self.bg_mask = self.bg_seg.get_mask()
        self.bg_mask = self.bg_mask | nonROI_mask
        self.bg_mask = self.bg_mask & (~self.human_mask)    #<- Trust human mask more
        # robot
        self.robot_seg.process(img)
        self.robot_mask = self.robot_seg.get_mask()
        self.robot_mask[self.human_mask] = False            #<- Trust the human mask and the bg mask more
        self.robot_mask[self.bg_mask] = False
        # puzzle
        self.puzzle_seg.set_detected_masks([self.bg_mask, self.human_mask, self.robot_mask])
        self.puzzle_seg.process(img)
        self.puzzle_mask = self.puzzle_seg.get_mask()
    
    def measure(self, img):
        raise NotImplementedError
    
    def predict():
        raise NotImplementedError
    
    def correct():
        raise NotImplementedError
    
    def adapt():
        raise NotImplementedError
    
    def nonROI(self):
        """
        This function encode the prior knowledge of which region is not of interest

        Current non-ROI: 
        1. Depth is zero, which indicate failure in depth capturing
        2. Height is too big (above 0.5 meter), which indicates non-table region

        @param[out] mask        The binary mask indicating non-ROI
        """
        assert (self.height_map is not None) and (self.depth is not None)

        mask = np.zeros_like(self.depth, dtype=bool) 
        # non-ROI 1
        mask[np.abs(self.depth) < 1e-3] = 1
        # non-ROI 2
        mask[self.height_map > 0.5] = 1

        return mask


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
            layer = cv2.warpPerspective(
                layer.astype(np.uint8), 
                self.params.BEV_trans_mat,
                (layer.shape[1], layer.shape[0])
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
            seg = eval("self."+layer_name+"_seg")
            if seg.tracker is not None:
                state = seg.tracker.getState()
                # Requires the shape (1, N, D). See:https://stackoverflow.com/questions/45817325/opencv-python-cv2-perspectivetransform
                state.tpt = cv2.perspectiveTransform(
                    state.tpt.T[np.newaxis, :, :],
                    self.params.BEV_trans_mat
                ).squeeze().T
                seg.tracker.displayState(state, ax)

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