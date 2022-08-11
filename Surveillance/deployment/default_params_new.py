"""

    @ brief         The default parameters for the Surveillance System, including the parameter settings and the
                    default post-processing methods

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
def get_params(layer, cfg):
    """Get the layer segmentation parameters

    Args:
        layer (str):        The desired layer name. ["human", "puzzle", "robot", "bg"]
        cfg (_type_):       The configuration parameters
    """
    if layer == "human":
        params = Human_Seg.Params(
            det_th=cfg.SceneSegment.humanGauss.varTh,
            postprocessor=lambda mask: \
                cv2.dilate(
                    mask.astype(np.uint8),
                    np.ones((10, 10), dtype=np.uint8),
                    1
                ).astype(bool)
        )
    elif layer == "bg":
        # parameters - tabletop
        params = Tabletop_Seg.Params_GMM(
            history=300,
            NMixtures=cfg.SceneSegment.bgGMM.gaussNum,
            varThreshold=cfg.SceneSegment.bgGMM.varTh,
            detectShadows=True,
            ShadowThreshold=cfg.SceneSegment.bgGMM.shadowTh,
            postprocessor=lambda mask: mask
        )
    elif layer == "robot":
        # parameters - robot
        params = Robot_Seg.Params()
    elif layer == "puzzle":
        # parameters - puzzle
        kernel = np.ones((15, 15), np.uint8)
        mask_proc_puzzle_seg = maskproc(
            maskproc.opening, (kernel,),
            maskproc.closing, (kernel,),
        )
        params = Puzzle_Seg.Params_Residual(
            # postprocessor=lambda mask: \
                # mask_proc_puzzle_seg.apply(mask.astype(bool))
        )
    else:
        raise NotImplementedError
    
    return params

def get_trackers(layer, cfg=None):
    """Get the tracker parameter settings

    Args:
        layer (str):        human or puzzle
        cfg ():             The configuration parameters
    """
    # trackers - human
    if layer == "human":
        tracker = centroid.centroid(
            params=centroid.Params(
                plotStyle="bo"
            )
        )
    elif layer == "puzzle":
        # trackers - puzzle
        tracker = mCentroid.centroidMulti(
            params=mCentroid.Params(
                plotStyle="rx"
            )
        )
    else:
        raise NotImplementedError

    return tracker

def get_ROI_mode(cfg):
    return cfg.SceneSegment.ROI.mode

