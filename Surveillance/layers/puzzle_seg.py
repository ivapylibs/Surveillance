#=============================== puzzle_seg ==============================
"""
  @brief          Puzzle segmenter to extract puzzle layer.

  @author         Yiye Chen.          yychen2019@gatech.edu
  @date           09/21/2021

"""
#=============================== puzzle_seg ==============================

# dependencies
from dataclasses import dataclass
import numpy as np

import detector.bgmodel.bgmodelGMM as bgmodelGMM
import detector.inImage as detector
import improcessor.basic as improcessor

import Surveillance.layers.base_fg as base_fg

#====================== puzzle_seg.Params_Residual =====================
"""
  @brief    Parameter class for Puzzle_Residual instances.
"""
@dataclass
class Params_Residual(base_fg.Params):
    def __post_init__(self):
        return super().__post_init__()

#=========================== Puzzle_Residual ===========================
#
"""
  @brief    Class to detect puzzle layer based on exclusion mask(s).

  The process of taking the residual is treated as the routine postprocess
"""

class Puzzle_Residual(base_fg.Base_fg):

    #=============================== init ==============================
    """
    @param[in]    theTracker      Tracker for puzzle pieces.
    @param[in]    trackFilter     Track filter for puzzle pieces.
    @param[in]    params          Parameter instance.
    """
    def __init__(self, theTracker=None, trackFilter=None, params:Params_Residual=Params_Residual()):

        # create a dummy detector that mark all pixels as the puzzle,
        # which will be corrected in the post_process
        theDetector = detector.inImage(
            processor=improcessor.basic(
                lambda img: np.ones_like(img[:, :, 0], dtype=bool),
                ()  #<- The dummy requires no params
            )
        )
        super().__init__(theDetector, theTracker=theTracker, trackFilter=trackFilter, params=params)

        # postprocess
        self.post_process_custom = params.postprocessor
        self.update_params("postprocessor", self.post_process)

        # cache the detection masks
        self.detected_masks = []

    #=========================== post_process ==========================
    """
    @brief      Define the initial post-processing routine.

    The post processing routine applies the user-customized postprocess
    after the initial post-processing step.

    The routine:
      (a) take the residual of the detection masks

    @param[in]  det_mask            The dummy detection result that is all True
    """
    def post_process(self, det_mask):

        final_mask = det_mask
        for mask in self.detected_masks: 
            final_mask = final_mask & ~(mask)

        # apply the customized postprocessor
        final_mask = self.post_process_custom(final_mask)

        return final_mask 

    #======================== update_postprocess =======================
    """
    @brief  Update the customized post-processor.

    @param[in]  postprocessor   The new custom post-processing routine.
    """
    def update_postprocess(self, postprocessor:callable):
        self.post_process_custom = postprocessor

    #======================== set_detected_masks =======================
    #
    # @brief    Pass along exclusion masks (not puzzle regions).
    #
    # @param[in]    masks   Set of masks.
    def set_detected_masks(self, masks):
        self.detected_masks = masks

    #============================= det_mask ============================
    """
    @brief  Overload the detection mask get function. 

    Suited for the inImage detector
    """
    def det_mask(self):
        return self.detector.Ip


#
#=============================== puzzle_seg ==============================
