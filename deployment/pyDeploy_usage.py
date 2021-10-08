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
from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

import Surveillance.utils.display as display
import Surveillance.layers.scene as scene 
import Surveillance.layers.human_seg as Human_Seg
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg

# == [1] Prepare the camera runner
d435_configs = d435.D435_Configs(
    W_dep=1280,
    H_dep=720,
    W_rgb=1920,
    H_rgb=1080
)

d435_starter = d435.D435_Runner(d435_configs)

# == [2] build a scene interpreter by running the calibration routine
fh = plt.figure()
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
    d435_starter.get_frames,
    d435_starter.intrinsic,
    rTh_high=1,
    rTh_low=0.02,
    hTracker=human_tracker,
    pTracker=puzzle_tracker,
    hParams=human_params,
    rParams=robot_Params,
    pParams=puzzle_params,
    bgParms=bg_seg_params,
    params=scene.Params(
        BEV_trans_mat=d435_starter.BEV_mat
    )
)

# == [3] Deploy
while(True):
    rgb, dep, status = d435_starter.get_frames()
    scene_interpreter.process_depth(dep)
    scene_interpreter.process(rgb)
    scene_interpreter.vis_scene(fh=fh)

