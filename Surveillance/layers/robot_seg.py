#=============================== robot_seg ===============================
##
# @package  Surveillance.layers.robot_seg
#
# @brief    OBSOLETE. The robot segmenters that aims to obtain the robot layer
#           mask and state.
#   
# @author:    Yiye Chen       yychen2019@gatech.edu
# @date:      07/30/2021
#
#=============================== robot_seg ===============================

import cv2
from dataclasses import dataclass
import copy
import numpy as np

import Surveillance.layers.base_fg as base_fg
from Surveillance.utils.height_estimate import HeightEstimator

import improcessor.basic as improcessor
import detector.inImage as detector

@dataclass
class Params(base_fg.Params):
    """
    @param  preprocessor        Executable on the input image
    @param  postprocessor       Executable on the detection mask to obtain the final layer mask
    """
    def __post_init__(self):
        return super().__post_init__()

class robot_inRange(base_fg.Base_fg):
    def __init__(self, low_th, high_th, tracker=None, trackFilter=None, \
                params:Params=Params()):
        """
        The inRange robot segmenter detects the segmenter with the assumption that 
        the depth of the robot falls within a certain range.

        @param[in]  low_th              float.The lower threshold for the inRange method
        @param[in]  high_th             float.The higher threshold for teh inRange method
        """


        # inRange detector
        Det = detector.inImage(
            improcessor.basic(cv2.inRange,(low_th,high_th))
        )

        super().__init__(Det, tracker, trackFilter, params=params)

        pass

    def det_mask(self):
        """
        Get the detection mask. 
        Based on the ivapylib/detector/testing/*_inRange.py, the in-range is constructed using the base class inImage,
        and the mask is get from self.Ip
        TODO: seems a little weird to me. I think there should be a generic get mask method
        """
        return self.detector.Ip
    
class robot_inRange_Height(robot_inRange):
    """
    The robot segmenter that uses the inRange segmentation on the height map

    The height based segmentation is treated as a routine post-process step.
    The customized postprocess passed through the params will be applied 
    after the routine post-process

    The height map can be obtained in two ways:
    1. Pass a heightEstimator to the constructor, and call process_depth function
        to obtrain the height
    2. Use the setHeight function to set the heigth directly

    NOTE: Now for sanity only allows one of the ways to be activated.
    (i.e.) If the heightEstimator is stored, then setHeight will be disabled
    If the heightEstimator is None, then process_depth will be disabled

    @param[in]  low_th              float.The lower threshold for the inRange method
    @param[in]  high_th             float.The higher threshold for teh inRange method
    @param[in]  theHeightEstimator  HeightEstimator. Default is None. Used to estimate
                                    height from the depth. If None, then requires set the 
                                    depth before processing each frame.
    """
    def __init__(self, low_th, high_th, theHeightEstimator:HeightEstimator=None, 
                tracker=None, trackFilter=None, params: Params=Params()):
        super().__init__(low_th, high_th, tracker=tracker, trackFilter=trackFilter, params=params)
 
        # height estimator
        self.height_estimator = theHeightEstimator
        self.height_map = None

        # parse out the customer post-processer and apply after the post_process routine
        self.post_process_custom = copy.deepcopy(params.postprocessor)

        # update the postprocessor to add in the routine postprocess
        self.update_params("postprocessor", self.post_process)
    
    def post_process(self, det_mask):
        """
        Define the whole post-process, which includes:
        1. Routine post-process (height based inRange segmentaiton)
        2. Customized post-process
        """

        # ===== [1] Routine post-process TODO
        # Assume the height map has been estimated
        assert self.height_map is not None, \
            "Please get the height map first, either by providing a HeightEstimator and call\
                process_depth function, or by update_height_map directly"
        
        # process of rgb image is meaningless, so simply overwrite
        self.detector.process(self.height_map)
        final_mask = self.detector.Ip.astype(bool)

        # ===== [2] Customized post-process
        final_mask = self.post_process_custom(final_mask)

        return final_mask 
    
    def process_depth(self, depth):
        """
        Process the depth
        """
        # for sanity
        assert self.height_estimator is not None, \
            "The height_estimator is None. Can not process depth. \
                Please add the heigth_estimator or use the setHeight to set the heigth directly"
        
        # TODO: get the height map
        self.height_map = np.abs(self.height_estimator.apply(depth))

    def update_height_map(self, height_map):
        """
        Set the cached height map
        When the heightEstimator is None, can use this to set the height map externally

        @param[in]  height_map              The height_map to store
        """
        self.height_map = height_map
