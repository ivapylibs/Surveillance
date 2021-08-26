
"""
 ============================== human_seg ===============================

    @brief             The human-segmentor for the layered approach
    
    @author:    Yiye Chen       yychen2019@gatech.edu
    @date:      07/29/2021

 ============================== human_seg ===============================
"""

from dataclasses import dataclass

import Surveillance.layers.base_fg as base_fg
from detector.fgmodel.targetSG import targetSG
from detector.fgmodel.targetSG import Params as targetSG_Params

@dataclass
class Params(base_fg.Params, targetSG_Params):
    """
    @param  preprocessor        Executable on the input image
    @param  postprocessor       Executable on the detection mask to obtain the final layer mask
    """
    # Any new params?
    def __post_init__(self):
        return super().__post_init__()

class Human_ColorSG(base_fg.Base_fg):
    def __init__(self, theDetector:targetSG, theTracker, trackFilter, params:Params):
        """
        The human segmentor with the single-Gaussian based color detector  
        """
        super().__init__(theDetector, theTracker, trackFilter, params)

    def det_mask(self):
        """
        det_mask only gets the detection mask without postprocess refinement
        Overwrite the det_mask function
        """
        if self.detector is None:
            return None
        else:
            return self.detector.getForeGround()

    @staticmethod 
    def buildFromImage(img, tracker=None, trackFilter=None, params:Params=Params()):
        """
        Return a Human_colorSG instance whose detector(targetSG) is built from the input image
        """
        detector = targetSG.buildFromImage(img, params=params)
        return Human_ColorSG(detector, tracker, trackFilter, params)


class Human_ColorSG_HeightInRange(Human_ColorSG):
    """
    The human detector that first use the Single Gaussian Color segmentation
    to segment the target color.
    Then the height in-range segmentation is used to segment out the whole human layer,
    where the height map is estimated from the camera intrinsics and the depth map

    This class wraps the functions in the testing/humanSG03.py
    """

    def __init__(self, theDetector: targetSG, theTracker, trackFilter, 
                params: Params):
        super().__init__(theDetector, theTracker, trackFilter, params)
        self.depth = None
        self.intrinsics = None

    def post_process(self, postprocess):
        """
        Define the post-process routine.
        Also allow users to add additional postprocess to the procedure.
        """
        pass

    def update_depth(self, depth):
        """
        Update the stored depth frame for the post process
        """
        self.depth = depth

    @staticmethod
    def buildFromImage(img_color, dep_height, intrinsics, tracker=None, \
        trackerFilter=None, params:Params=Params()):
        """
        Overwrite the base buildFromImage

        Now building a ColorSG_DepInRange human segmentor requires three things
        1. An image contraining the target color for the color model calibration
        2. Camera intrinsics for the height estimation from the depth image
        3. An depth map of the empty table surface for the height estimator calibration
        """
        pass
