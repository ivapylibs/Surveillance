
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

@dataclass
class Params(base_fg.Params):
    # Any new params?
    def __post_init__(self):
        return super().__post_init__()

class Human_ColorSG(base_fg.Base_fg):
    def __init__(self, theDetector:targetSG, theTracker, trackFilter, params:Params):
        """
        The human segmentor with the single-Gaussian based color detector  

        @param[in]  preprocessor        Executable on the input image
        @param[in]  postprocessor       Executable on the detection mask to obtain the final layer mask
        """
        super().__init__(theDetector, theTracker, trackFilter, params)

    def det_mask(self):
        """
        Overwrite the det_mask function
        """
        if self.detector is None:
            return None
        else:
            return self.detector.getForeGround()

    @staticmethod 
    def buildFromImage(img, tracker, trackFilter=None, params:Params=Params()):
        """
        Return a Human_colorSG instance whose detector(targetSG) is built from the input image
        """
        detector = targetSG.buildFromImage(img)
        return Human_ColorSG(detector, tracker, trackFilter, params)
