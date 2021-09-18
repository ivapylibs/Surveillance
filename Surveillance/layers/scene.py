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
import numpy as np
import matplotlib.pyplot as plt

# detectors
from Surveillance.utils.height_estimate import HeightEstimator
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.robot_seg import robot_inRange_Height
from detector.bgmodel.bgmodelGMM import bgmodelGMM_cv

# Params
from Surveillance.layers.human_seg import Params as Params_human
from Surveillance.layers.robot_seg import Params as Params_robot
from detector.bgmodel.bgmodelGMM import Params_cv as Params_bg

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
                bg_seg: bgmodelGMM_cv, 
                heightEstimator: HeightEstimator, 
                params: Params):
        self.params = params

        self.height_estimator = heightEstimator

        # segmenters
        self.human_seg = human_seg
        self.robot_seg = robot_seg
        self.bg_seg = bg_seg

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
        self.height_map = self.height_estimator.apply(depth)
        # update the height_map to those that requires
        self.human_seg.update_height_map(self.height_map)
        self.robot_seg.update_height_map(self.height_map)
    
    def process(self, img):
        """
        """
        # For now might only need to implement this one. 
        # Just let the detectors to process the image one-by-one

        # bg
        self.bg_seg.process(img)
        self.bg_mask = ~self.bg_seg.getForeground()
        # human
        self.human_seg.process(img)
        self.human_mask = self.human_seg.get_mask()
        # robot
        self.robot_seg.process(img)
        self.robot_mask = self.robot_seg.get_mask()
        # puzzle
        self.puzzle_mask = self.detect_puzzle_layer()
    
    def measure(self, img):
        raise NotImplementedError
    
    def predict():
        raise NotImplementedError
    
    def correct():
        raise NotImplementedError
    
    def adapt():
        raise NotImplementedError
    
    def detect_puzzle_layer(self):
        """
        Puzzle layer is assumed to be the remaining of the other layers
        So will use the self.bg_mask, human_mask, and robot_mask to determine the puzzle_mask
        """
        raise NotImplementedError

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
        
        raise NotImplementedError
        return None
    
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

    def vis_scene(self, 
                mask_only:List[bool]=[False, False, False, False], 
                BEV_rectify:List[bool]=[False, False, False, True]
    ):
        """
        Visualize four layers ["bg", "human", "robot", "puzzle"]
        """
        raise NotImplementedError


    @staticmethod
    def buildFromImages(self):
        """
        Build an SceneInterpreterV1 instance
        """
        raise NotImplementedError
        return None