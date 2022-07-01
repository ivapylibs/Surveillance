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
from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams

@dataclass
class Params(bParams):
    hand_radius: float = None      # If None, then will auto compute the radius

class HumanPuzzleSurveillance(BaseSurveillanceDeploy):
    """
    """

    def __init__(self, imgSource, scene_interpreter: scene.SceneInterpreterV1, params: Params = Params()) -> None:
        super().__init__(imgSource=imgSource, scene_interpreter=scene_interpreter, params=params)
       
        self.frame_writer_orig = frameWriter(
            dirname=self.params.save_dir,
            frame_name=self.params.save_name + "_original",
            path_idx_mode=True
        )

        self.img_BEV = None
        self.humanImg = None
        self.puzzleImg = None

        # the hand radius
        self.hand_radius = None         # The hand radius set by the user. If None, will automatically computed for each frame and stored in the self.hand_radius_adapt
        self.hand_radius_adapt = None

    def postprocess(self, rgb, dep):
        self.near_human_puzzle_idx = self.get_near_hand_puzzles(rgb, dep)
        

    def vis_results(self, rgb, dep):


        # the trackers
        hTracker_BEV = self.scene_interpreter.get_trackers("human", BEV_rectify=True)  # (2, 1)
        pTracker_BEV = self.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)  # (2, N)
        hand_mask = self.scene_interpreter.get_layer("human", mask_only=True, BEV_rectify=True)

        # The human + puzzle image
        img = self.scene_interpreter.get_layer("human", mask_only=False, BEV_rectify=True)[:, :, ::-1] \
                + self.scene_interpreter.get_layer("puzzle", mask_only=False, BEV_rectify=True)[:, :, ::-1]

        # Plot the marker for the near hand pieces
        if self.near_human_puzzle_idx is not None:
            # determine the hand range
            if self.hand_radius is None:
                r = self.hand_radius_adapt
            else:
                r = self.hand_radius
            # plot the hand range
            img = cv2.circle(img, hTracker_BEV.squeeze().astype(int), radius=int(self.hand_radius_adapt), color=(0, 0, 255),
                                thickness=10)
            # plot the puzzle markers
            for i in self.near_human_puzzle_idx:
                img = cv2.circle(img, pTracker_BEV[:, i].squeeze().astype(int), radius=20, color=(0, 255, 0),
                                    thickness=-1)
                        
        # visualize the scene interpreter result
        display.display_images_cv([self.puzzleImg[:,:,::-1], self.humanImg[:,:,::-1]], ratio=0.4)

        # visualize the near-hand puzzles
        text = "Near-hand puzzles, which is the puzzle pieces locate within a circle around the hand."
        display.display_images_cv([img], ratio=0.5, window_name=text)
        

    def save_data(self):
        return

    def get_near_hand_puzzles(self, rgb, dep):
        """Get the puzzle pieces that are possibly near to the hand
        
        It is done by drawing a circular region around the BEV_rectified hand centroid, 
        and then test whether the puzzle pieces locates within the region.

        The circular area is automatically determined by the distance between the fingure tip point and the centroid

        Args:
            rgb
            dep
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
            if self.hand_radius is None:
                hy, hx = np.where(hand_mask)
                fingertip_idx = np.argmax(hy)
                r = ((hx[fingertip_idx] - hTracker_BEV[0]) ** 2 + (hy[fingertip_idx] - hTracker_BEV[1]) ** 2) ** 0.5
                # distances = np.sum(
                #    (np.concatenate((hx[np.newaxis, :], hy[np.newaxis, :]), axis=0) - hTracker_BEV)**2,
                #    axis=0
                # )
                # r = np.amin(distances)
                self.hand_radius_adapt = r
            else:
                r = self.hand_radius
            #  get puzzle-to-human distances
            distances = np.sum(
                (pTracker_BEV - hTracker_BEV) ** 2,
                axis=0
            ) ** 0.5
            near_hand = distances < r
            idx = np.where(near_hand)[0]

        return idx



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
        reCalibrate=False,
        calib_data_save_dir=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cache_human_puzzle_2"
        )
    )

    # == [1] Prepare the camera runner & extrinsic calibrator
    human_puzzle_Surveillance = HumanPuzzleSurveillance.build(configs)

    # == [2] Deploy
    human_puzzle_Surveillance.run()
