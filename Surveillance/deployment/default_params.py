"""

    @ brief         The default parameters for the Surveillance System, including the settings and the post-processing methods

"""

import cv2
import numpy as np

from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

import Surveillance.layers.scene as scene
import Surveillance.layers.human_seg as Human_Seg
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg

# Defaults params
# parameters - human
HPARAMS = Human_Seg.Params(
    det_th=8,
    postprocessor=lambda mask: \
        cv2.dilate(
            mask.astype(np.uint8),
            np.ones((10, 10), dtype=np.uint8),
            1
        ).astype(bool)
)
# parameters - tabletop
BGPARMAS = Tabletop_Seg.Params_GMM(
    history=300,
    NMixtures=5,
    varThreshold=30.,
    detectShadows=True,
    ShadowThreshold=0.6,
    postprocessor=lambda mask: mask
)
# parameters - robot
ROBPARAMS = Robot_Seg.Params()
# parameters - puzzle
kernel = np.ones((15, 15), np.uint8)
mask_proc_puzzle_seg = maskproc(
    maskproc.opening, (kernel,),
    maskproc.closing, (kernel,),
)
PPARAMS = Puzzle_Seg.Params_Residual(
    postprocessor=lambda mask: \
        mask_proc_puzzle_seg.apply(mask.astype(bool))
)

# trackers - human
HTRACKER = centroid.centroid(
    params=centroid.Params(
        plotStyle="bo"
    )
)

# trackers - puzzle
PTRACKER = mCentroid.centroidMulti(
    params=mCentroid.Params(
        plotStyle="rx"
    )
)

# Non - ROI
def NONROI_FUN(H, W, top_ratio=0.2, down_ratio=0.1):
    nonROI_region = np.zeros((H, W), dtype=bool)
    nonROI_region[:int(H*top_ratio), int(H*down_ratio):] = 1
    return nonROI_region