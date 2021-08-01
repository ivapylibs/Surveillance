"""
===================================== robot_seg ===================================

    @brief          The robot segmenters that aims to obtain the robot layer mask
                    and state

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           07/30/2021

===================================== robot_seg ===================================
"""
import cv2
from dataclasses import dataclass

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
    
