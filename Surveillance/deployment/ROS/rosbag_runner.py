"""

    @brief          The test script that run the system on a rosbag file,
                    include building the system and test the system.

    @author         Yiye Chen,          yychen2019@gatech.edu
                    Yunzhi Lin,         yunzhi.lin@gatech.edu
    @date           02/25/2022

"""
import shutil
import numpy as np
import os
import subprocess
import yaml
import threading
import time
import cv2
import matplotlib.pyplot as plt
import argparse
import copy
from pathlib import Path

import rospy
import rosgraph
import rosbag

# Utils
from ROSWrapper.subscribers.Images_sub import Images_sub
from camera.utils.display import display_images_cv, display_rgb_dep_cv

# Surveillance system
from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams
from Surveillance.deployment.utils import terminate_process_and_children

# puzzle stuff
from puzzle.runner import RealSolver, ParamRunner
from puzzle.piece.template import Template, PieceStatus

# activity
from Surveillance.activities.state import StateEstimator


# configs
test_rgb_topic = "/test_rgb"
test_dep_topic = "/test_depth"

# prepare
lock = threading.Lock()
timestamp_ending = None
roscore_proc = None

# To be built
call_back_num = 0

def get_args():
    parser = argparse.ArgumentParser(description="Surveillance runner on the pre-saved rosbag file")
    parser.add_argument("--fDir", type=str, default="./", \
                        help="The folder's name.")

    parser.add_argument("--rosbag_name", type=str, default="data/Testing/Yiye/move_state_test_withPuzzle.bag", \
                        help="The rosbag file name.")
    parser.add_argument("--real_time", action='store_true', \
                        help="Whether to run the system for real-time or just rosbag playback instead.")
    parser.add_argument("--force_restart", action='store_true', \
                        help="Whether force to restart the roscore.")
    parser.add_argument("--display", action='store_false', \
                        help="Whether to display.")
    parser.add_argument("--puzzle_solver", action='store_true', \
                        help="Whether to apply puzzle_solver.")
    parser.add_argument("--state_analysis", action='store_true', \
                        help="Whether to apply the state analysis.")
    parser.add_argument("--verbose", action='store_true', \
                        help="Whether to debug the system.")
    parser.add_argument("--save_to_file", action='store_true', \
                        help="Whether save to files, the default file location is the same as the rosbag or realtime.")

    args = parser.parse_args()
    return args

class ImageListener:
    def __init__(self, opt):

        self.opt = opt

        # Initialize the saving folder
        if not self.opt.real_time:
            self.opt.save_folder = Path(self.opt.rosbag_name).stem
        else:
            self.opt.save_folder = 'realtime'

        # Clear up the space
        if os.path.exists(self.opt.save_folder):
            shutil.rmtree(self.opt.save_folder)
        os.makedirs(self.opt.save_folder, exist_ok=True)

        # Data captured
        self.RGB_np = None
        self.D_np = None

        self.rgb_frame_stamp = None
        self.rgb_frame_stamp_prev = None

        # Initialize a node
        rospy.init_node("test_surveillance_on_rosbag")

        # Build up the surveillance system from the rosbag
        configs_surv = bParams(
            visualize=False,
            ros_pub=False,
            # The calibration topics
            # deployment
            BEV_mat_topic="BEV_mat",
            intrinsic_topic="intrinsic",
            # scene interpreter
            empty_table_rgb_topic="empty_table_rgb",
            empty_table_dep_topic="empty_table_dep",
            glove_rgb_topic="glove_rgb",
            human_wave_rgb_topic="human_wave_rgb",
            human_wave_dep_topic="human_wave_dep",
            depth_scale_topic="depth_scale",
            # Postprocessing
            bound_limit = [300,300,400,0],
            mea_test_r = 100,  # @< The circle size in the postprocessing for the measured board
            mea_sol_r = 300,  # @< The circle size in the postprocessing for the solution board
            hand_radius = 200  # @< The hand radius set by the user.
        )
        self.surv = BaseSurveillanceDeploy.buildFromRosbag(rosbag_file, configs_surv)

        # Build up the puzzle solver
        configs_puzzleSolver = ParamRunner(
            areaThresholdLower=1000,
            areaThresholdUpper=10000,
            pieceConstructor=Template,
            tauDist=100
        )
        self.puzzleSolver = RealSolver(configs_puzzleSolver)

        # State analysis
        self.state_parser = StateEstimator(
            signal_number=1,
            signal_names=["location"],
            state_number=1,
            state_names=["Move"],
            move_th=20
        ) 

        # Initialize a subscriber
        Images_sub([test_rgb_topic, test_dep_topic], callback_np=self.callback_rgbd)

        print("Initialization ready, waiting for the data...")

    def callback_rgbd(self, arg_list):

        if self.opt.verbose:
            print("Get to the callback")

        RGB_np = arg_list[0]
        D_np = arg_list[1]
        rgb_frame_stamp = arg_list[2].to_sec()

        # np.integer includes both signed and unsigned, whereas the np.int only includes signed
        if np.issubdtype(D_np.dtype, np.integer):
            if self.opt.verbose:
                print("converting the depth scale")
            D_np = D_np.astype(np.float32) * self.surv.depth_scale

        with lock:
            self.RGB_np = RGB_np.copy()
            self.D_np = D_np.copy()
            self.rgb_frame_stamp = copy.deepcopy(rgb_frame_stamp)

    def run_system(self):

        with lock:

            global call_back_num

            if self.RGB_np is None:
                if self.opt.verbose:
                    print('No data')
                return

            RGB_np = self.RGB_np.copy()
            D_np = self.D_np.copy()
            rgb_frame_stamp = copy.deepcopy(self.rgb_frame_stamp)

            # Skip images with the same timestamp as the previous one
            if rgb_frame_stamp != None and self.rgb_frame_stamp_prev == rgb_frame_stamp:
                # if self.opt.verbose:
                #     print('Same timestamp')
                return
            else:
                self.rgb_frame_stamp_prev = rgb_frame_stamp

            if self.opt.verbose:
                print("Running the Surveillance on the test data")

            self.surv.process(RGB_np, D_np)

            # For demo
            humanImg = self.surv.humanImg
            puzzleImg = self.surv.puzzleImg
            humanMask = self.surv.humanMask
            nearHandImg = self.surv.humanAndhumanImg

            # For further processing
            postImg = self.surv.meaBoardImg
            hTracker = self.surv.hTracker

            # For near-human-hand puzzle pieces.
            # @note there may be false positives
            hTracker_BEV = self.surv.scene_interpreter.get_trackers("human", BEV_rectify=True)  # (2, 1)
            # pTracker_BEV = self.surv.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)  # (2, N)
            near_human_puzzle_idx = self.surv.near_human_puzzle_idx # @< pTracker_BEV is the trackpointers of all the pieces.
            print('Idx from puzzle solver:', near_human_puzzle_idx) # @< The index of the pTracker_BEV that is near the human hand

            if self.opt.display:
                # Display
                display_images_cv([self.RGB_np[:, :, ::-1]], ratio=0.5, window_name="Source RGB")
                # display_images_cv([humanImg[:, :, ::-1], puzzleImg[:, :, ::-1]], ratio=0.5, window_name="Separate layers")
                # display_images_cv([postImg[:, :, ::-1]], ratio=0.5, window_name="meaBoardImg")
                display_images_cv([nearHandImg[:, :, ::-1]], ratio=0.5, window_name="nearHandImg")

                cv2.waitKey(1)

            if self.opt.save_to_file:
                # Save for debug
                cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_rgb.png'), self.RGB_np[:, :, ::-1])
                cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_hand.png'), humanImg[:, :, ::-1])
                cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_handMask.png'),
                            humanMask)
                cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_puzzle.png'), postImg[:, :, ::-1])
                cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_nearHand.png'), nearHandImg[:, :, ::-1])

            if self.opt.puzzle_solver:
                # Work on the puzzle pieces

                # Todo: Currently, initialize the SolBoard with the very first frame.
                # We can hack it with something outside
                if call_back_num==0:
                    self.puzzleSolver.setSolBoard(postImg)

                # Plan not used yet
                plan, id_dict = self.puzzleSolver.process(postImg, hTracker_BEV)

                # @note there may be false negatives
                print('ID from puzzle solver:', id_dict)

                if self.opt.display:
                    # Display
                    display_images_cv([self.puzzleSolver.bMeasImage[:, :, ::-1]], ratio=1, window_name="Measured board")
                    # display_images_cv([self.puzzleSolver.bTrackImage[:, :, ::-1]], ratio=1, window_name="Tracking board")
                    # display_images_cv([self.puzzleSolver.bSolImage[:, :, ::-1]], ratio=1, window_name="Solution board")

                    cv2.waitKey(1)

                if self.opt.save_to_file:
                    # Save for debug
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_bMeas.png'), self.puzzleSolver.bMeasImage[:, :, ::-1])
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_bTrack.png'), self.puzzleSolver.bTrackImage[:, :, ::-1])

                # # Compute progress
                # thePercent = self.puzzleSolver.progress()
                # print(f"Progress: {thePercent}")

            # Hand moving states
            # NOTE: here I only invoke the state parser when the hand is detected (hTracker is not None)
            # This might cause unsynchronization with the puzzle states.
            # So might need to set the self.move_state to indicator value when the hand is not detected.
            if self.opt.state_analysis and hTracker is not None:
                # get the tracker
                self.state_parser.process([hTracker])
                self.state_parser.visualize(RGB_np, window_name="State")

                # NOTE: The moving state is obtained here.
                # The return is supposed to be of the shape (N_state, ), where N_state is the number of states, 
                # since it was designed to include extraction of all the states.
                # Since the puzzle states is implemented elsewhere, the N_state is 1, hence index [0]
                self.move_state = self.state_parser.get_states()[0]
                # print(self.move_state)

            call_back_num += 1
            print("The processed test frame number: {} \n\n".format(call_back_num))

        # Only applied when working on rosbag playback
        if self.opt.real_time is False:

            global timestamp_ending
            global roscore_proc

            # # Debug only
            if self.opt.verbose:
                print('Current:', rgb_frame_stamp)
                print('Last:', timestamp_ending)

            # We ignore the last 2 seconds
            if timestamp_ending is not None and abs(rgb_frame_stamp - timestamp_ending) < 2:
                print('Shut down the system.')
                rospy.signal_shutdown('Finished')
                # Stop the roscore if started from the script
                if roscore_proc is not None:
                    terminate_process_and_children(roscore_proc)

if __name__ == "__main__":

    # Parse arguments
    args = get_args()
    rosbag_file = os.path.join(args.fDir, args.rosbag_name)

    # Local configuration for debug

    args.save_to_file = True
    # args.verbose = True
    # args.display = False
    args.puzzle_solver = True
    # args.state_analysis = True
    args.force_restart = True


    if args.force_restart:
        subprocess.call(['killall rosmaster'], shell=True)

    # Start the roscore if not enabled
    if not rosgraph.is_master_online():
        roscore_proc = subprocess.Popen(['roscore'])
        # wait for a second to start completely
        time.sleep(1)

    listener = ImageListener(args)

    plt.ion()
    if args.real_time is False:
        # Get basic info from the rosbag
        info_dict = yaml.safe_load(
            subprocess.Popen(['rosbag', 'info', '--yaml', rosbag_file], stdout=subprocess.PIPE).communicate()[0])
        timestamp_ending = info_dict['end']

        print('Playback the rosbag recordings')

        # Need to start later for initialization
        # May need to slow down the publication otherwise the subscriber won't be able to catch it
        command = "rosbag play {} -d 2 -r 1 --topic {} {}".format(
           rosbag_file, test_rgb_topic, test_dep_topic)

        try:
           # Be careful with subprocess, pycharm needs to start from the right terminal environment (.sh instead of shortcut)
           # https://stackoverflow.com/a/3478415
           # We do not want to block the process
           subprocess.Popen(command, shell=True)
        except:
           print("Cannot execute the bash command: \n {}".format(command))
           exit()

    while not rospy.is_shutdown():
        listener.run_system()
