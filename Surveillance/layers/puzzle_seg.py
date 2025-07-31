#=============================== puzzle_seg ==============================
##
# @package  Surveillance.layers.puzzle_seg
#
# @brief    OBSOLETE. The puzzle segmenters that aims to extract the puzzle
#           layer
#   
# @author:    Yiye Chen       yychen2019@gatech.edu
# @date           09/21/2021
#
#=============================== puzzle_seg ==============================

# dependencies
from dataclasses import dataclass
import numpy as np

import detector.bgmodel.bgmodelGMM as bgmodelGMM
import detector.inImage as detector
import improcessor.basic as improcessor

import Surveillance.layers.base_fg as base_fg

#params definition
@dataclass
class Params_Residual(base_fg.Params):
    """
    @param preprocess: the preprocess of the input image
    @param postprocess: post process of the detected layer mask (after the routine postprocess)
    """
    # Any new parameters?
    #
    def __post_init__(self):
        return super().__post_init__()

# classes definition
class Puzzle_Residual(base_fg.Base_fg):
    """
    This class detect the puzzle layer as the residual of the other masks

    The process of taking residual is treated as the roution postprocess
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

    def post_process(self, det_mask):
        """
        Define the routine post-process
        And apply the user-customize postprocess after the routine the procedure.

        The routine:
        (a) take the residual of the detection masks

        @param[in]  det_mask            The dummy detection result that is all True
        """
        final_mask = det_mask
        for mask in self.detected_masks: 
            final_mask = final_mask & ~(mask)

        # apply the customized postprocessor
        final_mask = self.post_process_custom(final_mask)

        return final_mask 

    def update_postprocess(self, postprocessor:callable):
        """
        update the postprocessor, which is the customized one after applying the height-based segmentation process.
        """
        self.post_process_custom = postprocessor

    def set_detected_masks(self, masks):
        self.detected_masks = masks

    def det_mask(self):
        """
        Overwrite the detection mask get function. 

        Suited for the inImage detector
        """
        return self.detector.Ip
