"""

    @brief          The test of the Surveillance system on human puzzle playing

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           10/18/2021

"""

from dataclasses import dataclass
from posixpath import dirname
import cv2
import numpy as np
import os
import sys
import time

import camera.d435.d435_runner as d435
from camera.extrinsic.aruco import CtoW_Calibrator_aruco
from camera.utils.utils import BEV_rectify_aruco
from camera.utils.writer import frameWriter
import camera.utils.display as display 

from improcessor.mask import mask as maskproc
import trackpointer.centroid as centroid
import trackpointer.centroidMulti as mCentroid

import Surveillance.layers.scene as scene
import Surveillance.layers.human_seg as Human_Seg
import Surveillance.layers.robot_seg as Robot_Seg
import Surveillance.layers.tabletop_seg as Tabletop_Seg
import Surveillance.layers.puzzle_seg as Puzzle_Seg

deployPath = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(deployPath)
from Base import BaseSurveillanceDeploy
from Base import Params as bParams

@dataclass
class Params(bParams):
    near_hand_radius: float = None      # If None, then will auto compute the radius
    def __post_init__(self):
        return super().__post_init__()

class HumanPuzzleSurveillance(BaseSurveillanceDeploy):
    """
    """

    def __init__(self, imgSource, scene_interpreter: scene.SceneInterpreterV1, params: Params = Params()) -> None:
        super().__init__()
       

        self.frame_writer_orig = frameWriter(
            dirname=self.params.save_dir,
            frame_name=self.params.save_name + "_original",
            path_idx_mode=True
        )


        self.img_BEV = None
        self.humanImg = None
        self.puzzleImg = None

    def postprocess(self, rgb, dep):
        self.near_human_puzzle_idx = self.get_near_hand_puzzles(rgb, dep, vis=True)
        

    def vis_results(self, rgb, dep, ra):
        # the trackers
        hTracker_BEV = self.scene_interpreter.get_trackers("human", BEV_rectify=True)  # (2, 1)
        pTracker_BEV = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)  # (2, N)
        hand_mask = self.scene_interpreter.get_layer("human", mask_only=True, BEV_rectify=True)
        # The human + puzzle image
        img = self.scene_interpreter.get_layer("human", mask_only=False, BEV_rectify=True)[:, :, ::-1] \
                + self.scene_interpreter.get_layer("puzzle", mask_only=False, BEV_rectify=True)[:, :, ::-1]
        # if nothing is near hand
        if self.near_human_puzzle_idx is not None:
            img = cv2.circle(img, hTracker_BEV.squeeze().astype(int), radius=int(r), color=(1, 0, 255),
                                thickness=10)
            for i in self.near_human_puzzle_idx:
                img = cv2.circle(img, pTracker_BEV[:, i].squeeze().astype(int), radius=20, color=(0, 255, 0),
                                    thickness=-1)
        text = "Near-hand puzzles, which is the puzzle pieces locate within a circle around the hand."
        if radius is not None:
            text = text + " The radius is auto-computed"
        display.display_images_cv([img], ratio=0.5, window_name=text)
        

    def save_data(self):
        return

    def get_near_hand_puzzles(self, rgb, dep, radius=None, vis=False):
        """Get the puzzle pieces that are possibly near to the hand
        
        It is done by drawing a circular region around the BEV_rectified hand centroid, 
        and then test whether the puzzle pieces locates within the region.

        The circular area is automatically determined by the distance between the fingure tip point and the centroid

        Args:
            rgb
            dep
            radius (int. Optional).     The radius of the near_hand region. Defaults to None, in which case will be auto-determined \
                by the furthest target hand color pixel from the centroid.
            vis (bool. Optional).  If set to True, then will visualize the hand + puzzle layer with the hand centroid, \
                circular region, and the high light of the possibly interacted puzzle pieces
        Returns:
            idx [np.ndarray].   The index of the puzzle pieces that is near to the hand. If none, then return None
        """
        hTracker_BEV = self.scene_interpreter.get_trackers("human", BEV_rectify=True)  # (2, 1)
        pTracker_BEV = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)  # (2, N)
        hand_mask = self.scene_interpreter.get_layer("human", mask_only=True, BEV_rectify=True)
        idx = np.empty((0))

        # if either hand or puzzle pieces are not presented, then return None
        if hTracker_BEV is None or pTracker_BEV is None or np.all(hand_mask == False):
            idx = None
        # otherwise
        else:
            # determine radius
            if radius is None:
                hy, hx = np.where(hand_mask)
                fingertip_idx = np.argmax(hy)
                r = ((hx[fingertip_idx] - hTracker_BEV[0]) ** 2 + (hy[fingertip_idx] - hTracker_BEV[1]) ** 2) ** 0.5
                # distances = np.sum(
                #    (np.concatenate((hx[np.newaxis, :], hy[np.newaxis, :]), axis=0) - hTracker_BEV)**2,
                #    axis=0
                # )
                # r = np.amin(distances)
            else:
                r = radius
            #  get puzzle-to-human distances
            distances = np.sum(
                (pTracker_BEV - hTracker_BEV) ** 2,
                axis=0
            ) ** 0.5
            near_hand = distances < r
            idx = np.where(near_hand)[0]
            # print(idx)

        # visualization
        if vis:
            # The human + puzzle image
            img = self.scene_interpreter.get_layer("human", mask_only=False, BEV_rectify=True)[:, :, ::-1] \
                  + self.scene_interpreter.get_layer("puzzle", mask_only=False, BEV_rectify=True)[:, :, ::-1]
            # if nothing is near hand
            if idx is not None:
                img = cv2.circle(img, hTracker_BEV.squeeze().astype(int), radius=int(r), color=(1, 0, 255),
                                 thickness=10)
                for i in idx:
                    img = cv2.circle(img, pTracker_BEV[:, i].squeeze().astype(int), radius=20, color=(0, 255, 0),
                                     thickness=-1)
            text = "Near-hand puzzles, which is the puzzle pieces locate within a circle around the hand."
            if radius is not None:
                text = text + " The radius is auto-computed"
            display.display_images_cv([img], ratio=0.5, window_name=text)
        return idx

    def _get_measure_board(self):
        """
        Compare to the puzzle segmentation mask, the measure board carves out a larger circular area
        around each puzzle piece region to get high recall.
        """
        if self.params.board_type == "test":
            meaBoardMask, meaBoardImg = self._get_measure_board_test()
        elif self.params.board_type == "solution":
            meaBoardMask, meaBoardImg = self._get_measure_board_sol()
        return meaBoardMask, meaBoardImg

    def _get_measure_board_test(self):
        puzzle_seg_mask = self.scene_interpreter.get_layer("puzzle", mask_only=True, BEV_rectify=True)
        puzzle_tpt = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)

        # initialize the measure board mask  
        meaBoardMask = np.zeros_like(puzzle_seg_mask, dtype=bool)
        # if some puzzle piece are tracked
        if puzzle_tpt is not None:
            # get the centroid mask. Note the tpt is in opencv coordinate system
            centroids = puzzle_tpt.astype(int)
            cols = centroids[0, :]
            rows = centroids[1, :]
            rows[rows >= self.img_BEV.shape[0]] = self.img_BEV.shape[0] - 1
            cols[cols >= self.img_BEV.shape[1]] = self.img_BEV.shape[1] - 1
            meaBoardMask[rows, cols] = 1

            # dilate with circle
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.params.mea_test_r, self.params.mea_test_r))
            mask_proc = maskproc(
                maskproc.dilate, (kernel,)
            )
            meaBoardMask = mask_proc.apply(meaBoardMask)

        # finally obtain the meaBoardImg
        meaBoardImg = meaBoardMask[:, :, np.newaxis].astype(np.uint8) * self.img_BEV
        return meaBoardMask, meaBoardImg

    def _get_measure_board_sol(self):
        # get the puzzle segmentation mask, trackpointers, and the img_BEV
        puzzle_seg_mask = self.scene_interpreter.get_layer("puzzle", mask_only=True, BEV_rectify=True)
        puzzle_tpt = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)

        # initialize the measure board mask  
        meaBoardMask = np.zeros_like(puzzle_seg_mask, dtype=bool)
        # get the centroid
        x, y = np.where(puzzle_seg_mask)
        if x.size != 0:
            meaBoardMask[int(np.mean(x)), int(np.mean(y))] = 1
            # dilate with circle
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.params.mea_sol_r, self.params.mea_sol_r))
            mask_proc = maskproc(
                maskproc.dilate, (kernel,)
            )
            meaBoardMask = mask_proc.apply(meaBoardMask)
        # finally obtain the meaBoardImg
        meaBoardImg = meaBoardMask[:, :, np.newaxis].astype(np.uint8) * self.img_BEV
        return meaBoardMask, meaBoardImg

    @staticmethod
    def build(params: Params = Params()):
        # the cache folder for the data
        fDir = os.path.dirname(
            os.path.realpath(__file__)
        )
        cache_dir = os.path.join(
            fDir,
            "cache_human_puzzle"
        )
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        cache_dir = os.path.join(
            fDir,
            "cache_human_puzzle/" + params.bg_color
        )
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        # camera runner
        d435_configs = d435.D435_Configs(
            W_dep=848,
            H_dep=480,
            W_color=1920,
            H_color=1080,
            exposure=100,
            gain=55
        )

        d435_starter = d435.D435_Runner(d435_configs)
        intrinsic = np.array(
            [[1.38106177e+03, 0.00000000e+00, 9.78223145e+02],
             [0.00000000e+00, 1.38116895e+03, 5.45521362e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        )

        # The aruco-based calibrator
        calibrator_CtoW = CtoW_Calibrator_aruco(
            d435_starter.intrinsic_mat,
            distCoeffs=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            markerLength_CL=params.markerLength,
            maxFrames=10,
            stabilize_version=True
        )

        # == [2] build a scene interpreter by running the calibration routine

        # Calibrate the extrinsic matrix
        BEV_mat_path = os.path.join(cache_dir, "BEV_mat.npz")

        # Check if the calibration file is existing
        if not os.path.exists(BEV_mat_path):
            print('No calibration file exists. Start calibration.')
            params.reCalibrate = True

        if params.reCalibrate:

            rgb, dep = display.wait_for_confirm(lambda: d435_starter.get_frames()[:2],
                                                color_type="rgb",
                                                ratio=0.5,
                                                instruction="Camera pose estimation: \n Please place the Aruco tag close to the base for the Extrinsic and Bird-eye-view(BEV) matrix calibration. \n Press \'c\' to start the process. \n Please remove the tag upon completion",
                                                )
            while not calibrator_CtoW.stable_status:
                rgb, dep, _ = d435_starter.get_frames()
                M_CL, corners_aruco, img_with_ext, status = calibrator_CtoW.process(rgb, dep)
                assert status, "The aruco tag can not be detected"
            # calibrate the BEV_mat
            topDown_image, BEV_mat = BEV_rectify_aruco(rgb, corners_aruco, target_pos="down", target_size=100,
                                                       mode="full")
            # save
            np.savez(
                BEV_mat_path,
                BEV_mat=BEV_mat
            )
        else:
            print('Load the saved calibration files.')
            BEV_mat = np.load(BEV_mat_path, allow_pickle=True)["BEV_mat"]

        # parameters - human
        human_params = Human_Seg.Params(
            det_th=8,
            postprocessor=lambda mask: \
                cv2.dilate(
                    mask.astype(np.uint8),
                    np.ones((10, 10), dtype=np.uint8),
                    1
                ).astype(bool)
        )
        # parameters - tabletop
        bg_seg_params = Tabletop_Seg.Params_GMM(
            history=300,
            NMixtures=5,
            varThreshold=30.,
            detectShadows=True,
            ShadowThreshold=0.6,
            postprocessor=lambda mask: mask
        )
        # parameters - robot
        robot_Params = Robot_Seg.Params()
        # parameters - puzzle
        kernel = np.ones((15, 15), np.uint8)
        mask_proc_puzzle_seg = maskproc(
            maskproc.opening, (kernel,),
            maskproc.closing, (kernel,),
        )
        puzzle_params = Puzzle_Seg.Params_Residual(
            postprocessor=lambda mask: \
                mask_proc_puzzle_seg.apply(mask.astype(bool))
        )

        # trackers - human
        human_tracker = centroid.centroid(
            params=centroid.Params(
                plotStyle="bo"
            )
        )

        # trackers - puzzle
        puzzle_tracker = mCentroid.centroidMulti(
            params=mCentroid.Params(
                plotStyle="rx"
            )
        )

        # run the calibration routine
        scene_interpreter = scene.SceneInterpreterV1.buildFromSource(
            lambda: d435_starter.get_frames()[:2],
            # d435_starter.intrinsic_mat,
            intrinsic,
            rTh_high=1,
            rTh_low=0.02,
            hTracker=human_tracker,
            pTracker=puzzle_tracker,
            hParams=human_params,
            rParams=robot_Params,
            pParams=puzzle_params,
            bgParams=bg_seg_params,
            params=scene.Params(
                BEV_trans_mat=BEV_mat
            ),
            reCalibrate=params.reCalibrate,
            cache_dir=cache_dir
        )

        return HumanPuzzleSurveillance(d435_starter.get_frames, scene_interpreter, params)


if __name__ == "__main__":
    fDir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.dirname(
        os.path.dirname(fDir)
    )
    # save_dir = os.path.join(save_dir, "data/puzzle_solver_black")
    save_dir = os.path.join(save_dir, "data/temp")

    # == [0] Configs
    configs = Params(
        markerLength=0.08,
        save_dir=save_dir,
        save_name="GTSolBoard",
        bg_color="black",  # black or white
        reCalibrate=True,
        board_type="test",  # test or solution
        mea_test_r=125,
        mea_sol_r=250
    )

    # == [1] Prepare the camera runner & extrinsic calibrator
    human_puzzle_Surveillance = HumanPuzzleSurveillance.build(configs)

    # == [2] Deploy
    human_puzzle_Surveillance.run()
