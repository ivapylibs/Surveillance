"""
 ============================== base ===============================

    @brief              The base class for the foreground layer segmentor in the 
                        layered perception approach
    
    @author:    Yiye Chen       yychen2019@gatech.edu
    @date:      07/29/2021

 ============================== base ===============================
"""


from dataclasses import dataclass
import Surveillance.layers.base as base

@dataclass
class Params(base.Params):
    # Any additional parameters?
    #
    """
    @param preprocess: the preprocess of the input image
    @param postprocess: post process of the detected layer mask
    """
    def __post_init__(self):
        return super().__post_init__()

class Base_fg(base.Base):
    def __init__(self, theDetector, theTracker, trackFilter, params:Params):
        """
        Base class for the foreground segmentation in the layered approach

        The processing pipeline:
        preprocess -> detect -> postprocess -> track -> trackfilter

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
    