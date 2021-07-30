
"""
 ============================== human_seg ===============================

    @brief             The human-segmentor for the layered approach
    
    @author:    Yiye Chen       yychen2019@gatech.edu
    @date:      07/29/2021

 ============================== human_seg ===============================
"""

from Surveillance.layers.base_fg import Base_fg
from detector.fgmodel.targetSG import targetSG

class Human_ColorSG(Base_fg):
    def __init__(self, theDetector:targetSG, theTracker, trackFilter, **kwargs):
        """
        The human segmentor with the single-Gaussian based color detector  

        @param[in]  preprocessor        Executable on the input image
        @param[in]  postprocessor       Executable on the layer mask
        """
        super().__init__(theDetector, theTracker, trackFilter, **kwargs)

    def det_mask(self):
        """
        Overwrite the det_mask function
        """
        if self.detector is None:
            return None
        else:
            return self.detector.getForeGround()

    @staticmethod 
    def buildFromImage(img, tracker, trackFilter=None, **kwargs):
        """
        Return a Human_colorSG instance whose detector(targetSG) is built from the input image
        """
        detector = targetSG.buildFromImage(img)
        return Human_ColorSG(detector, tracker, trackFilter, **kwargs)
