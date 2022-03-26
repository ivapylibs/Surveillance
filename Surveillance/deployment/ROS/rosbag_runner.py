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

# ROS
import rospy
import rosgraph
import rosbag
from std_msgs.msg import UInt8

# Utils
from ROSWrapper.subscribers.Images_sub import Images_sub
from camera.utils.display import display_images_cv, display_rgb_dep_cv
from Surveillance.deployment.utils import display_option_convert

# Surveillance system
from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams
from Surveillance.deployment.utils import terminate_process_and_children
from Surveillance.deployment.activity_record import ActDecoder

# puzzle stuff
from puzzle.runner import RealSolver, ParamRunner
from puzzle.piece.template import Template, PieceStatus

# activity
from Surveillance.activity.state import StateEstimator

# configs
test_rgb_topic = "/test_rgb"
test_dep_topic = "/test_depth"
test_activity_topic = "/test_activity"

# preparation
lock = threading.Lock()
timestamp_ending = None
roscore_proc = None

# To be built
call_back_num = 0

def get_args():
    parser = argparse.ArgumentParser(description="Surveillance runner on the pre-saved rosbag file")
    parser.add_argument("--fDir", type=str, default="./", \
                        help="The folder's name.")
    parser.add_argument("--rosbag_name", type=str, default="data/Testing/Yunzhi/activity_simple.bag", \
                        help="The rosbag file name.")
    parser.add_argument("--real_time", action='store_true', \
                        help="Whether to run the system for real-time or just rosbag playback instead.")
    parser.add_argument("--force_restart", action='store_true', \
                        help="Whether force to restart the roscore.")
    parser.add_argument("--display", default=1, \
                        help="0/000000: No display;"
                             "1/000001: source input;"
                             "2/000010: hand;"
                             "4/000100: robot;"
                             "8/001000: puzzle;"
                             "16/010000: postprocessing;"
                             "32/100000: puzzle board;"
                             "You can use decimal or binary as the input.")
    parser.add_argument("--puzzle_solver", action='store_true', \
                        help="Whether to apply puzzle_solver.")
    parser.add_argument("--state_analysis", action='store_true', \
                        help="Whether to apply the state analysis. Display is automatically enabled.")
    parser.add_argument("--near_hand_demo", action='store_true', \
                        help="Whether to visualize near hand pieces.")
    parser.add_argument("--verbose", action='store_true', \
                        help="Whether to debug the system.")
    parser.add_argument("--save_to_file", action='store_true', \
                        help="Whether save to files, the default file location is the same as the rosbag or realtime.")

    args = parser.parse_args()

    return args

class ImageListener:
    def __init__(self, opt):

        self.opt = opt

        # Convert display code
        self.opt.display = display_option_convert(self.opt.display)

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
            bound_limit = [300,300,400,50], # @< The ignored region area.
            mea_test_r = 100,  # @< The circle size in the postprocessing for the measured board.
            mea_sol_r = 300,  # @< The circle size in the postprocessing for the solution board.
            hand_radius = 200  # @< The hand radius set by the user.
        )

        # build the surveillance deployer
        self.surv = BaseSurveillanceDeploy.buildFromRosbag(rosbag_file, configs_surv)

        # Build up the puzzle solver
        configs_puzzleSolver = ParamRunner(
            areaThresholdLower=1000,
            areaThresholdUpper=10000,
            pieceConstructor=Template,
            tauDist=100 # @< The radius distance determining the near-by pieces.
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

        # Initialize the activity label subscriber, decoder, and the label storage if needed
        self.activity_label = None
        if args.read_activity:
            rospy.Subscriber(test_activity_topic, UInt8, callback=self.callback_activity, queue_size=1)
            self.act_decoder = ActDecoder()

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
    
    def callback_activity(self, key_msg):
        # decode the activity
        key = chr(key_msg.data)
        self.act_decoder.decode(key)

        # store the activity label
        self.activity_label = self.act_decoder.get_activity()

    def run_system(self):

        with lock:

            global call_back_num

            if self.RGB_np is None:
                if self.opt.verbose:
                    print('No data')
                return

            rgb_frame_stamp = copy.deepcopy(self.rgb_frame_stamp)

            # Skip images with the same timestamp as the previous one
            if rgb_frame_stamp != None and self.rgb_frame_stamp_prev == rgb_frame_stamp:
                # if self.opt.verbose:
                #     print('Same timestamp')
                return
            else:
                self.rgb_frame_stamp_prev = rgb_frame_stamp
                RGB_np = self.RGB_np.copy()
                D_np = self.D_np.copy()
                activity = copy.deepcopy(self.activity_label)

            if self.opt.verbose:
                print("Running the Surveillance on the test data")

            self.surv.process(RGB_np, D_np)

            # For demo
            humanImg = self.surv.humanImg
            robotImg = self.surv.robotImg
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

            # Display
            if self.opt.display[0]:
                if self.activity_label is not None:
                    RGB_np = cv2.putText(np.float32(RGB_np), self.activity_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                         2.0, (255, 0, 0), 5)
                    RGB_np = np.uint8(RGB_np)
                display_images_cv([RGB_np[:, :, ::-1]], ratio=0.5, window_name="Source RGB")
            if self.opt.display[1]:
                display_images_cv([humanImg[:, :, ::-1]], ratio=0.5, window_name="Hand layer")
            if self.opt.display[2]:
                display_images_cv([robotImg[:, :, ::-1]], ratio=0.5, window_name="Robot layer")
            if self.opt.display[3]:
                display_images_cv([puzzleImg[:, :, ::-1]], ratio=0.5, window_name="Puzzle layer")
            if self.opt.display[4]:
                display_images_cv([postImg[:, :, ::-1]], ratio=0.5, window_name="Postprocessing (Input to the puzzle solver)")
            if self.opt.near_hand_demo:
                self.surv.vis_near_hand_puzzles()
                display_images_cv([nearHandImg[:, :, ::-1]], ratio=0.5, window_name="nearHandImg")

            # If there is at least one display command
            if any(self.opt.display) or self.opt.near_hand_demo:
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
                # We assume SolBoard is perfect (all the pieces have been recognized successfully)
                # We can hack it with something outside
                if call_back_num == 0:
                    self.puzzleSolver.setSolBoard(postImg)

                # Plan not used yet
                plan, id_dict = self.puzzleSolver.process(postImg, hTracker_BEV)

                # @note there may be false negatives
                print('ID from puzzle solver:', id_dict)

                if self.opt.display[5]:
                    # Display
                    display_images_cv([self.puzzleSolver.bMeasImage[:, :, ::-1], self.puzzleSolver.bTrackImage[:, :, ::-1], self.puzzleSolver.bSolImage[:, :, ::-1]],
                                      ratio=0.5, window_name="Measured/Tracking/Solution board")

                    cv2.waitKey(1)

                if self.opt.save_to_file:
                    # Save for debug
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_bMeas.png'), self.puzzleSolver.bMeasImage[:, :, ::-1])
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_num).zfill(4)}_bTrack.png'), self.puzzleSolver.bTrackImage[:, :, ::-1])

                # Compute progress
                # Note that the solution board should be correct, otherwise it will fail.
                try:
                    thePercent = self.puzzleSolver.progress()
                    print(f"Progress: {thePercent}")
                except:
                    print('Double check the solution board to make it right.')

            # Hand moving states
            # Todo: here I only invoke the state parser when the hand is detected (hTracker is not None)
            # This might cause non-synchronization with the puzzle states.
            # So might need to set the self.move_state to indicator value when the hand is not detected.
            if hTracker is None:
                self.move_state = 0
            elif self.opt.state_analysis and hTracker is not None:
                # get the tracker
                self.state_parser.process([hTracker])
                self.state_parser.visualize(RGB_np, window_name="State")

                # NOTE: The moving state is obtained here.
                # The return is supposed to be of the shape (N_state, ), where N_state is the number of states, 
                # since it was designed to include extraction of all the states.
                # Since the puzzle states is implemented elsewhere, the N_state is 1, hence index [0]
                self.move_state = self.state_parser.get_states()[0]

            print(self.move_state)

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

    # args.save_to_file = True
    # args.verbose = True
    args.display = '110001'
    args.puzzle_solver = True
    args.state_analysis = True
    # args.force_restart = True

    # update the args about the existence of the activity topic
    bag = rosbag.Bag(rosbag_file)

    if len(list(bag.read_messages(test_activity_topic))) != 0:
        args.read_activity = True
    else:
        args.read_activity = False

    if args.force_restart:
        subprocess.call(['killall rosbag'], shell=True)
        subprocess.call(['killall rosmaster'], shell=True)

    # Start the roscore if not enabled
    if not rosgraph.is_master_online():
        roscore_proc = subprocess.Popen(['roscore'], shell=True)
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
        command = "rosbag play {} -d 2 -r 1 -s 1 --topic {} {} {}".format(
           rosbag_file, test_rgb_topic, test_dep_topic, test_activity_topic)

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
