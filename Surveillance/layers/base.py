"""
 ============================== base ===============================

    @brief              The base class for the layer segmentor in the 
                        layered perception approach
    
    @author:    Yiye Chen       yychen2019@gatech.edu
    @date:      07/29/2021

 ============================== base ===============================
"""

import perceiver.simple as simple
from detector.inImage import inImage
import matplotlib.pyplot as plt

from dataclasses import dataclass

doNothing_func = lambda x: x

@dataclass
class Params(simple.Params):
    preprocessor: callable = doNothing_func          # the callable on the input data
    postprocessor: callable = doNothing_func        # the callable on the mask from the detector, result of which will be the layer mask
    def __post_init__(self):
        super().__init__()


class Base(simple.simple):
    def __init__(self, theDetector, theTracker, trackFilter, params:Params):
        """
        Base class for the layer segmentation approach

        Build upon the the simple detector -> tracker -> trackfilter pipeline

        Now building a base classs of the following process pipeline:
        preprocess -> detect -> postprocess -> track -> trackfilter

        where:
        - preprocess: the preprocess of the input image
        - Postprocess: post process of the detected layer mask
        - params: need to has the field of the Params class 

        """
        if theDetector is not None:
            # to make sure the detector has the desired API
            assert isinstance(theDetector, inImage)

        super().__init__(theDetector, theTracker, trackFilter, params)

        # the ultimate goal of the layer segmenter is to obtain the mask of the layer and a tracking state (e.g. trackpointer)
        self.layer_mask_det = None          # the mask obtained from the detector
        self.layer_mask = None              # the mask after post-processing as the final answer
        self.layer_state = None             # the tracking state

        # store the params
        self.params = params
    
    def get_mask(self):
        return self.layer_mask
    
    def get_state(self):
        return self.layer_state

    def measure(self, I):
        """
        Set a common processing pipeline?

        But different detector or tracker will generate different result name. 
        e.g. the layer mask should be obtained from fg_detector.getForeGround() and bg_detector.getBackground() separetely
        similar for the trackers 

        Might be better off defining the pipeline separately for different subclass of segmentor? Or just make up some default?

        TODO: here requires the tracker instance to have the process & getstate API.
        might be better to also limit the input to some base tracker class with those APIs?
        """
        # --[1] Preprocess
        Ip = self.params.preprocessor(I)

        # --[2] process
        if self.detector is not None:
            self.detector.process(Ip)
            self.layer_mask_det = self.det_mask()
        else:
            self.layer_mask_det = None

        # --[3] Postprocess
        self.layer_mask = self.params.postprocessor(self.layer_mask_det)

        # --[4] Track state
        if self.tracker is not None:
            self.tracker.process(self.layer_mask)
            self.layer_state = self.tracker.getstate()
        else:
            self.layer_state = None

    def det_mask(self):
        """
        Make the default getting the foregound mask

        To be overwritten for any specific detector type
        """
        if self.detector is None:
            return None
        else:
            raise NotImplementedError("Baseclass does not make assumption on how the detection mask can be get.\
                Need to be overwritten by child classes")

    def draw_layer(self, img=None):
        """
        Visualize the layer result

        @ param[in] img         The input image. Default is None. If not None, then will crop the layer mask area and show.
                                If None, then will only plot the binary mask
        """
        ax = plt.gca()

        # draw the layer
        mask = self.get_mask()
        if img is None:
            ax.imshow(mask, "Greys")
        else:
            if len(img.shape) == 3:
                ax.imshow(img * mask[:,:,None])
            elif len(img.shape) == 2:
                ax.imshow(img * mask)
        
        # draw the tracker state. 
        # TODO: here requires the tracker instance to have the displayState method 
        if self.tracker is not None:
            self.tracker.displayState()
    
    