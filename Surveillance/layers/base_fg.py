#================================ Base_fg ================================
"""
  @brief    Base class for foreground layer segmentor in layered
            perception approach
    
  @author   Yiye Chen       yychen2019@gatech.edu
  @date     07/29/2021

 ============================== base ===============================
"""


from dataclasses import dataclass
import Surveillance.layers.base as base

#============================ base_fg.Params ===========================
"""
  @brief    Parameter class for Base_fg instance.
"""
@dataclass
class Params(base.Params):
    def __post_init__(self):
        return super().__post_init__()


#=============================== Base_fg ===============================
"""
  @brief    Base class for foreground segmentation in layered approach.

  The processing pipeline goes as follows:
    preprocess -> detect -> postprocess -> track -> trackfilter

  Some of these steps can be no-ops. Simplicity/complexity is determined
  by how the individual steps are defined. The base class needs for it
  all to be overloaded/specified.
"""
class Base_fg(base.Base):
    #=============================== init ==============================
    """
    @brief  Constructor for Base_fg class.

    @param[in]    theDetector     Base detector.
    @param[in]    theTracker      Base tracker.
    @param[in]    trackFilter     Track filter (smoothing /data association).
    @param[in]    params          Parameter instance.
    """
    def __init__(self, theDetector, theTracker, trackFilter, params:Params):
        super().__init__(theDetector, theTracker, trackFilter, params)


    #============================= det_mask ============================
    """
    @brief  Get the current foreground mask.
    """
    def det_mask(self):
        if self.detector is None:
            return None
        else:
            return self.detector.getForeGround()


#
#================================ Base_fg ================================
