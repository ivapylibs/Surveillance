
"""
 ============================== human_seg ===============================

    @brief             The human-segmentor for the layered approach
    
    @author:    Yiye Chen       yychen2019@gatech.edu
    @date:      07/29/2021

 ============================== human_seg ===============================
"""

from dataclasses import dataclass
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label

import Surveillance.layers.base_fg as base_fg
from detector.fgmodel.targetSG import targetSG
from detector.fgmodel.targetSG import Params as targetSG_Params
from improcessor.mask import mask as maskproc
from Surveillance.utils.height_estimate import HeightEstimator

# define global symbol for util functions
getLargestCC = lambda mask:maskproc.getLargestCC(mask)

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
    The human detector based on the Single Gaussian Color segmentation
    It adds the postprocess routine that use the depth map to estimate the height,
    and then perform in-range segmentation to get the human layer mask.

    The postprocessor passed to the class constructor will be performed after the heightInRange process 
    """

    def __init__(self, theDetector: targetSG, theHeightEstimator: HeightEstimator = None, 
                theTracker = None, trackFilter = None, 
                params: Params = Params()):
        super().__init__(theDetector, theTracker, trackFilter, params)

        self.height_estimator = theHeightEstimator
        self.depth = None
        self.intrinsics = None

        self.height_map = None 

        # parse out the customer post-processer and apply after the post_process routine
        self.post_process_custom = copy.deepcopy(params.postprocessor)

        # update the postprocessor to be the routine postprocess + customized postprocess
        self.update_params("postprocessor", self.post_process)

    def post_process(self, det_mask):
        """
        Define the post-process routine.
        Also allow users to add additional postprocess to the procedure.

        The function to:
        (a) get the height map from the depth map
        (b) perform thresholding on the height map and find the connected component to the largest CC of the init_mask
        (c) assuming the hand is reaching out from the top of the image frame, remove all pixels so far below the init_mask as outlier
        """

        # get the height_map
        if self.height_estimator is not None:
            self.height_map = np.abs(self.height_estimator.apply(self.depth))
        else:
            # allow set the heigth map externally
            assert self.height_map is not None, "Both the height map and the height_estimator is None. \
                Please set either a height estimator or the height map"

        # if the detection returns no result, then also return no result
        if np.all(det_mask == 0):
            return np.zeros_like(det_mask, dtype=bool)

        # threshold
        init_height = self.height_map[det_mask]

        # TODO: histogram shows that there will always be some values close to zero.
        # lets take off the values too close to the ground
        #plt.figure(10)
        #plt.hist(init_height, bins=50, density=True)
        if np.all(init_height <= 0.01):
            return np.zeros_like(det_mask, dtype=bool)
        init_height = init_height[init_height>0.01]
        low = np.amin(init_height)
        mask = self.height_map > low 

        # Connected components of mask 
        labels_mask, num_labels = ndi.label(mask)
        # Check which connected components contain pixels from mask_high.
        sums = ndi.sum(det_mask, labels_mask, np.arange(num_labels + 1))
        connected_to_max_init = sums == max(sums)   # by take only the max, the non-largest connected component of the init_mask will be ignored
        max_connect_mask = connected_to_max_init[labels_mask]

        # remove pixels so far below the init mask
        cols_init = np.where(det_mask==1)[0]
        col_max = np.amax(cols_init)
        final_mask = copy.deepcopy(max_connect_mask)
        final_mask[col_max+10:] = 0

        # also assume the detection mask would be all positive
        final_mask = final_mask | getLargestCC(det_mask)

        # apply the customized postprocessor
        final_mask = self.post_process_custom(final_mask)

        return final_mask 

    def update_depth(self, depth):
        """
        Update the stored depth frame for the post process
        """
        self.depth = depth

    def update_postprocess(self, postprocessor:callable):
        """
        update the postprocessor, which is the customized one after applying the height-based segmentation process.
        """
        self.post_process_custom = postprocessor
    
    def update_height_map(self, height_map):
        self.height_map = height_map

    def draw_height(self, ax=None):
        """
        Draw the estimated height map
        """
        assert self.height_map is not None, \
            "please apply the height estimation first before visualizing it"

        if ax is None:
            ax = plt.gca()

        ax.imshow(self.height_map)

    @staticmethod
    def buildFromImage(img_color, dep_height=None, intrinsics=None, tracker=None, \
        trackerFilter=None, params:Params=Params()):
        """
        Overwrite the base buildFromImage

        Now building a ColorSG_HeightInRange human segmentor requires three things
        1. An image contraining the target color for the color model calibration
        2. Camera intrinsics for the height estimation from the depth image
        3. An depth map of the empty table surface for the height estimator calibration

        @param[in]  img_color           The image containing the target color
        @param[in]  dep_height          The depth map of the empty table for height estimator calibration. 
                                        Default is None, meaning a height estimator won't be built. Will need to set height before processing
        @param[in]  intrinsic           THe camera intrinsic fo the height estimator. 
                                        Default is None, meaning that a height estimator won't be build. Will need to set height before processing
        """
        detector = targetSG.buildFromImage(img_color, params=params)
        if (dep_height is not None) and (intrinsics is not None):
            height_estimator = HeightEstimator(intrinsic=intrinsics)
            height_estimator.calibrate(dep_height)
        else:
            height_estimator = None
        return Human_ColorSG_HeightInRange(detector, height_estimator, tracker, trackerFilter, params)
    
    @staticmethod
    def buildImgDiff(img_bg, img_fg, dep_height=None, intrinsics=None, tracker=None, \
        trackerFilter=None, params:Params=Params()):
        """Create a ColorSG_HeightInRange human segmenter by using the image difference method
        to automatically extract the target color samples and optionally using the empty table 
        depth data to calibrate a height estimator

        Args:
            img_bg ([type]): [description]
            img_fg ([type]): [description]
            dep_height (np.ndarray, (H, W), optional): The depth map of the empty table for height estimator calibration. \
                Defaults to None, which means will not create a height estimator and will rely externel feeding of the heigth info
            intrinsics (np.ndarray, (3, 3), optional): THe camera intrinsic for the height estimator. \
                Default to None, assuming that a height estimator won't be build.
            tracker (trackpointer, optional): Trackerpointer detector. Defaults to None.
            trackerFilter (trackerFilter, optional): Defaults to None.
            params (human_seg.Params, optional): segmenter parameters. Defaults to Params().

        Returns:
            human_segmenter [human_seg.Human_ColorSG_HeightInRange]: A ColorSG_HeightInRange segmenter instance
        """

        detector = targetSG.buildImgDiff(img_bg, img_fg, params=params)
        if (dep_height is not None) and (intrinsics is not None):
            height_estimator = HeightEstimator(intrinsic=intrinsics)
            height_estimator.calibrate(dep_height)
        else:
            height_estimator = None
        return Human_ColorSG_HeightInRange(detector, height_estimator, tracker, trackerFilter, params)
