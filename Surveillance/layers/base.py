"""
 ============================== base ===============================

    @brief              The base class for the layer segmentor in the 
                        layered perception approach
    
    @author:    Yiye Chen       yychen2019@gatech.edu
    @date:      07/29/2021

 ============================== base ===============================
"""

from perceiver.simple import simple
import matplotlib.pyplot as plt

class Base(simple):
    def __init__(self, theDetector, theTracker, trackFilter, **kwargs):
        """
        Base class for the layer segmentation approach

        Build upon the the simple detector -> tracker -> trackfilter pipeline

        Now building a base classs of the following process pipeline:
        preprocess -> detect -> postprocess -> track -> trackfilter

        where:
        - preprocess: the preprocess of the input image
        - Postprocess: post process of the detected layer mask

        """
        super().__init__(theDetector, theTracker, trackFilter, None, **kwargs)

        # the ultimate goal of the layer segmenter is to obtain the mask of the layer and a tracking state (e.g. trackpointer)
        self.layer_mask_det = None          # the mask obtained from the detector
        self.layer_mask = None              # the mask after post-processing as the final answer
        self.layer_state = None             # the tracking state

        # parse the mask exclusion, preprocessor, and postprocessor. Default if do nothing for the processors
        self.preprocessor = self._defaultIfMissiong(kwargs, "preprocessor", lambda x:x)
        self.postprocessor = self._defaultIfMissiong(kwargs, "postprocessor", lambda x:x)
    
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
        """
        # --[1] Preprocess
        Ip = self.preprocessor(I)

        # --[2] process
        if self.detector is not None:
            self.detector.process(Ip)
            self.layer_mask_det = self.det_mask()
        else:
            self.layer_mask_det = None

        # --[3] Postprocess
        self.layer_mask = self.postprocessor(self.layer_mask_det)

        # --[4] Track state
        if self.tracker is not None:
            self.tracker.process(self.layer_mask)
            self.layer_state = self.tracker.getstate()
        else:
            self.layer_state = None
        pass

    def det_mask(self):
        """
        Make the default getting the foregound mask

        To be overwritten for any specific detector type
        """
        if self.detector is None:
            return None
        else:
            # just make up a general API. will be Overwritten anyway
            return self.detector.getMask()

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
        
        # draw the tracker state
        if self.tracker is not None:
            self.tracker.displayState()
        
    
    
    def _defaultIfMissiong(self, dict: dict, key, default_val=None):
        if key in dict.keys() and dict[key] is None:
            return dict[key]
        else:
            return default_val
