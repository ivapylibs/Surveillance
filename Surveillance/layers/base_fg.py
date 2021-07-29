"""
 ============================== base ===============================

    @brief              The base class for the foreground layer segmentor in the 
                        layered perception approach
    
    @author:    Yiye Chen       yychen2019@gatech.edu
    @date:      07/29/2021

 ============================== base ===============================
"""

from Surveillance.layers.base import Base

class Base_fg(Base):
    def __init__(self, theDetector, theTracker, trackFilter, **kwargs):
        """
        Base class for the foreground segmentation in the layered approach

        The processing pipeline:
        preprocess -> detect -> postprocess -> track -> trackfilter

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
    