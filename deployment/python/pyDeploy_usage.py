"""

    @brief          The basic usage of the sceneInterpreter V1.0 builder for deployment

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           10/08/2021

"""

# == [0] Prepare the dependencies
import matplotlib.pyplot as plt
import cv2
import numpy as np

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco

from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

import Surveillance.utils.display as display
import Surveillance.layers.scene as scene 
import Surveillance.layers.human_seg as Human_Seg
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg

# == [1] Prepare the camera runner & extrinsic calibrator

# camera runner
d435_configs = d435.D435_Configs(
    W_dep=1280,
    H_dep=720,
    W_color=1920,
    H_color=1080
)

d435_starter = d435.D435_Runner(d435_configs)

# The aruco-based calibrator
calibrator_CtoW = CtoW_Calibrator_aruco(
    d435_starter.intrinsic_mat,
    distCoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    markerLength_CL = 0.067,
    maxFrames = 10,
    flag_vis_extrinsic = True,
    flag_print_MCL = True,
    stabilize_version = True
)

# == [2] build a scene interpreter by running the calibration routine
print("Calibrating the scene interpreter, please wait...")

# parameters - human
human_params = Human_Seg.Params(
    det_th=8,
    postprocessor= lambda mask:\
        cv2.dilate(
            mask.astype(np.uint8),
            np.ones((10,10), dtype=np.uint8),
            1
        ).astype(bool)
)
# parameters - tabletop
bg_seg_params = Tabletop_Seg.Params_GMM(
    history=300,
    NMixtures=5,
    varThreshold=15.,
    detectShadows=True,
    ShadowThreshold=0.55,
    postprocessor=lambda mask: mask
)
# parameters - robot
robot_Params = Robot_Seg.Params()
# parameters - puzzle
kernel= np.ones((9,9), np.uint8)
mask_proc = maskproc(
    maskproc.opening, (kernel, ),
    maskproc.closing, (kernel, ),
)
puzzle_params = Puzzle_Seg.Params_Residual(
    postprocessor=lambda mask: \
       mask_proc.apply(mask.astype(bool)) 
)

# trackers - human
human_tracker = centroid.centroid(
        params=centroid.Params(
            plotStyle="bo"
        )
    )

# trackers - puzzle
puzzle_tracker=mCentroid.centroidMulti(
        params=mCentroid.Params(
            plotStyle="rx"
        )
    )

# run the calibration routine
scene_interpreter = scene.SceneInterpreterV1.buildFromSource(
    lambda: d435_starter.get_frames()[:2],
    d435_starter.intrinsic_mat,
    rTh_high=1,
    rTh_low=0.02,
    hTracker=human_tracker,
    pTracker=puzzle_tracker,
    hParams=human_params,
    rParams=robot_Params,
    pParams=puzzle_params,
    bgParms=bg_seg_params,
    params=scene.Params(
        #BEV_trans_mat=d435_starter.BEV_mat
        BEV_trans_mat=None
    )
)

# == [3] Deploy
while(True):
    rgb, dep, status = d435_starter.get_frames()
    scene_interpreter.process_depth(dep)
    scene_interpreter.process(rgb)
    #scene_interpreter.vis_scene(fh=fh)

    layer = scene_interpreter.get_layer("puzzle", BEV_rectify=False)
    cv2.imshow("puzzle", layer[:,:,::-1])
    opKey = cv2.waitKey(1)
    if opKey == "q":
        break 

