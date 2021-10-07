"""
===================================== tabletop_seg ===================================

    @brief          The tabletop segmenters that aims to extract the tabletop layer
                    as the background

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           09/21/2021

===================================== tabletop_seg ===================================
"""

# dependencies
from dataclasses import dataclass
from cv2 import mean
import matplotlib.pyplot as plt
import detector.bgmodel.bgmodelGMM as bgmodelGMM
from matplotlib.pyplot import table

import Surveillance.layers.base_bg as base_bg

#params definition
@dataclass
class Params_GMM(base_bg.Params, bgmodelGMM.Params_cv):
    """
    @param preprocess: the preprocess of the input image
    @param postprocess: post process of the detected layer mask
    """
    # Any new parameters?
    #
    def __post_init__(self):
        return super().__post_init__()

# classes definition
class tabletop_GMM(base_bg.Base_bg):
    """
    The background layer segmenter based on the Gaussian Mixture model(GMM)

    @param[in]  theDetector         detector.bgmodel.bgmodelGMM.bgmodelGMM_cv instance. The detector based on the GMM
    @param[in]  params              tabletop_seg.Params_GMM instance.
                                    The parameters related to the segmenter, including the preprocessor, postprocess, etc.
    """
    def __init__(self, theDetector:bgmodelGMM.bgmodelGMM_cv, params:Params_GMM) -> None:
        super().__init__(theDetector, None, None, params)

    def calibrate(self, I):
        """
        Calibrate the GMM model parameter
        """
        # enable adapt
        self.detector.doAdapt = True

        # process the frame
        self.detector.process(I)

        # disable adapt
        self.detector.doAdapt = False
    
    def calibrate_from_source(self, source, fh=None):
        """The interface for calibrating the model parameter from a data source.
        The process would be as following:
        1. The functions displays the stream of rgb and depth from the data source
        2. When the user press any key to confirm "ready", then the calibration start
        3. When the user press any key to confirm "complete", then the calibration ends.
        4. The function ends

        Args:
            source (Any): Any data source that can get rgb and depth data by: \
                rgb, depth = source()
        """
        if fh is None:
            fh = plt.figure()
        


    @staticmethod
    def build(mParams: bgmodelGMM.Params_cv, params:Params_GMM()):
        """
        Build a tabletop_GMM instance

        @param[in]  mParams         The model parameters related to the Gaussian Mixture Model
        """
        detector_GMM = bgmodelGMM.bgmodelGMM_cv(mParams)
        return tabletop_GMM(detector_GMM, params)