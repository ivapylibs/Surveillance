"""

    @ brief         The default parameters for the Surveillance System, including the settings and the
    post-processing methods

    @author:        Yiye Chen,          yychen2019@gatech.edu
    @date:          02/26/2022

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
    varThreshold=150.,
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

# rgb preprocessing function 
def PREPROCESS_RGB(rgb):
    return rgb

# depth preprocessing function
import matplotlib.pyplot as plt
def PREPROCESS_DEPTH(depth):
    return depth
    #plt.ioff()

    # get the zero value map
    depth_missing = (depth == 0)
    depth_missing = depth_missing.astype(np.uint8)
    #plt.figure(1)
    #plt.imshow(depth)
    #plt.figure(2)
    #plt.imshow(depth_missing)

    #### Try the mean filter
    k_size = 5
    kernel = np.ones((k_size,k_size),np.float32)/(k_size**2)
    depth = cv2.filter2D(depth,-1,kernel)

    #### Try fill with the left most closest non-zero value
    depth_missing = (depth == 0)
    depth_missing = depth_missing.astype(np.uint8)
    rows, cols = np.nonzero(depth_missing)
    for row, col in zip(rows, cols):
        col_fill = col
        while(True):
            col_fill -= 1
            if col_fill < 0:
                break
            if depth[row, col_fill] != 0:
                depth[row, col] = depth[row, col_fill]
                break

    ##### Try to fill the zero value with the closest non-zero value - TOO slow...
    ## calculate the closest non-zero pixel index 
    ## NOTE: said that the setting blow gives "fast, coarse" estimation.
    ## source: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042
    ## The meaning of the labels: non-zero pixels and the closest zero pixels will share the same label
    #dist, labels = cv2.distanceTransformWithLabels(depth_missing, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

    ## get the non-zero pixel(pixel with 0 depth) and fill using the labels
    #rows, cols = np.nonzero(depth_missing)
    #for row, col in zip(rows, cols):
    #    label = labels[row, col]
    #    depth[labels==label] = np.max(depth[labels==label])
    
    #plt.figure(3)
    #plt.imshow(depth)
    #plt.show()
    
    return depth

# Non - ROI
def NONROI_FUN(H, W, top_ratio=0.2, down_ratio=0.1):
    nonROI_region = np.zeros((H, W), dtype=bool)
    nonROI_region[:int(H*top_ratio), int(H*down_ratio):] = 1
    return nonROI_region


# Activity codebook