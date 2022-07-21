#================================== base =================================
"""
  @brief    Base perceiver class for the layer segmentor in the layered
            perception approach. 
    
  @author   Yiye Chen       yychen2019@gatech.edu
  @date     07/29/2021

"""
#================================== base =================================

import perceiver.simple as simple
from detector.inImage import inImage
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, fields

doNothing_func = lambda x: x

#============================= base.Params =============================
"""
  @brief    Parameter class for Base perceiver instance.
    
  @param    preprocess      pre-processor of input image
  @param    postprocess     post-processor of detected layer mask
"""
@dataclass
class Params(simple.Params):
    preprocessor:   callable = doNothing_func # callable on input data
    postprocessor:  callable = doNothing_func # callable on det mask,
                                              #  output is the layer mask
    def __post_init__(self):
        super().__init__()

#================================= Base ================================
"""
  @brief    Base class for the layer segmentation approach

  Build upon the the simple detector -> tracker -> trackfilter pipeline

  Now building a base classs of the following process pipeline:
    preprocess -> detect -> postprocess -> track -> trackfilter
"""
class Base(simple.simple):
    #=============================== init ==============================
    """
    @brief  Constructor for Base_fg class.

    @param[in]    theDetector     Base detector.
    @param[in]    theTracker      Base tracker.
    @param[in]    trackFilter     Track filter (smoothing /data association).
    @param[in]    params          Parameter instance.
    """
    def __init__(self, theDetector, theTracker, trackFilter, params:Params):
        if theDetector is not None:
            # to make sure the detector has the desired API
            assert isinstance(theDetector, inImage)

        super().__init__(theDetector, theTracker, trackFilter, params)

        # the ultimate goal of the layer segmenter is to obtain the mask
        # of the layer and a tracking state (e.g. trackpointer)
        self.layer_mask_det = None  # mask obtained from detector
        self.layer_mask     = None  # mask after post-processing, final answer
        self.layer_state    = None  # the track state

        # store params
        self.params = params

    #========================== update_params ==========================
    """
    @brief  Update a parameter

    @param[in]  name    Name of the parameter.
    @param[in]  val     Value of the parameter.
    """
    def update_params(self, name, val):
        if name in [f.name for f in fields(self.params)]:
            setattr(self.params, name, val)
        else:
            assert False, "The query name - {} - can not be found. \
                The stored parameters are: {}".format(name, self.params)
    
    #============================= get_mask ============================
    """
    @brief  Get the current layer mask.
    """
    def get_mask(self):
        return self.layer_mask
    
    #============================ get_state ============================
    """
    @brief  Get the current track state.
    """
    def get_state(self):
        return self.layer_state

    #============================= measure =============================
    """
    @brief  Take in current input and generate measurement from data.

    Set a common processing pipeline?

    But different detector or tracker will generate different result name. 
    e.g. the layer mask should be obtained from fg_detector.getForeGround() 
    and bg_detector.getBackground() separetely similar for the trackers 

    Might be better off defining the pipeline separately for different
    subclass of segmentor? Or just make up some default?

    @todo here requires the tracker instance to have the process &
            getstate API.  might be better to also limit the input to
            some base tracker class with those APIs?
    """
    def measure(self, I):
        #--[1] Preprocess
        Ip = self.params.preprocessor(I)

        #--[2] process
        if self.detector is not None:
            self.detector.process(Ip)
            self.layer_mask_det = self.det_mask()
        else:
            self.layer_mask_det = None

        #--[3] Postprocess
        self.layer_mask = self.params.postprocessor(self.layer_mask_det)

        #--[4] Track state
        if self.tracker is not None:
            self.tracker.process(self.layer_mask)
            self.layer_state = self.tracker.getState()
        else:
            self.layer_state = None

    #============================= det_mask ============================
    """
    @brief  Get the current foreground mask.

    Expect to be overloaded for specialized detectors.
    """
    def det_mask(self):
        if self.detector is None:
            return None
        else:
            raise NotImplementedError(
              "Baseclass does not make assumption on how the detection mask \
              can be gotten. Need to be overloaded by child class(es).")

    #============================ draw_layer ===========================
    """
    @brief  Visualize the layer result

    @param[in]  img         Input image. Default: None. 
                              If not None, crops layer mask area and shows.
                              If None, plots the binary mask.
    @param[in] raw_detect   bool. Default: False. draw raw detected mask?
                              If drawn, will not display tracker state.
    """
    def draw_layer(self, img=None, raw_detect=False, ax=None):
        if ax is None:
            ax = plt.gca()

        # draw the layer
        if not raw_detect:
            mask = self.get_mask()
        else:
            mask = self.det_mask()

        if img is None:
            ax.imshow(mask, "Greys")
        else:
            if len(img.shape) == 3:
                ax.imshow((img * mask[:,:,None]).astype(np.uint8))
            elif len(img.shape) == 2:
                ax.imshow((img * mask).astype(np.uint8))
        
        # draw the tracker state. 
        # TODO: here requires tracker instance to have displayState method 
        if not raw_detect and self.tracker is not None:
            self.tracker.displayState(ax=ax)
    
