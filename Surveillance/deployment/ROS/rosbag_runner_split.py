#!/usr/bin/python3
"""
    @brief          The test script that run the surveillance system on a rosbag file,
                    include building the system and test the system.
                    Add more support to ROS, while the puzzle solver is split into a separate module.
                    The system will send/receive info from ROS topics, interacting with the puzzle solver ROS node.
                    We should do a similar thing for the activity analysis module.
    @author         Yiye Chen,          yychen2019@gatech.edu
                    Yunzhi Lin,         yunzhi.lin@gatech.edu
    @date           11/24/2022
"""
import glob
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
import pickle

# ROS
import rospy
import rosgraph
import rosbag
from std_msgs.msg import UInt8, String

from ROSWrapper.publishers.Matrix_pub import Matrix_pub
from ROSWrapper.publishers.Image_pub import Image_pub
from ROSWrapper.subscribers.String_sub import String_sub

# Utils
from ROSWrapper.subscribers.Images_sub import Images_sub
from camera.utils.display import display_images_cv, display_rgb_dep_cv
from Surveillance.deployment.utils import display_option_convert, calc_closest_factors

# Surveillance system
from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams
from Surveillance.deployment.utils import terminate_process_and_children
from Surveillance.deployment.activity_record import ActDecoder
from Surveillance.utils.imgs import draw_contour

# puzzle stuff
from puzzle.runner import RealSolver, ParamRunner
from puzzle.piece.template import Template, PieceStatus
from puzzle.utils.dataProcessing import convert_dict2ROS, convert_ROS2dict

# activity
from Surveillance.activity.state import StateEstimator
from Surveillance.activity.FSM import Pick, Place
from Surveillance.activity.utils import DynamicDisplay, ParamDynamicDisplay
from Surveillance.utils.configs import CfgNode_SurvRunner

# configs
# Read
test_rgb_topic = "/test_rgb"
test_dep_topic = "/test_depth"
test_activity_topic = "/test_activity"

# Publish
postImg_topic = "postImg"
visibleMask_topic = "visibleMask"
hTracker_BEV_topic = "hTracker_BEV"

# Subscribe, the name needs to be consistent with the one in the puzzle solver part
# See https://github.com/ivapylibs/puzzle_solver/tree/yunzhi/puzzle/testing/real_runnerROS.py
puzzle_solver_info_topic = "/puzzle_solver_info"
status_history_topic = "/status_history"
loc_history_topic = "/loc_history"

# @note Not that important in this module, just for display, maybe add later
bMeasImage_topic = "/bMeasImage"
bTrackImage_topic = "/bTrackImage"
bTrackImage_SolID_topic = "/bTrackImage_SolID"


# preparation
lock = threading.Lock()

timestamp_beginning = None
timestamp_ending = None
roscore_proc = None

# To be built
call_back_id = 0


def get_args():
    parser = argparse.ArgumentParser(description="Surveillance runner on the pre-saved rosbag file")
    parser.add_argument("--fDir", type=str, default="./",
                        help="The folder's name.")
    parser.add_argument("--yfile", default=None, type=str,  # type=lambda s: [item for item in s.split(',')],
                        help="The yaml configuration files in addition to the default yaml file. "
                             "It can overwrite the default configurations or add new parameters."
                             "See the ./config/default.yaml for the default parameters"
                        )
    parser.add_argument("--rosbag_name", type=str,
                        default="data/Testing/Adan/data_2022-05-06-11-09-27_Heather_Cupcake_Compressed.bag", \
                        help="The rosbag file name.")
    parser.add_argument("--real_time", action='store_true',
                        help="Whether to run the system for real-time or just rosbag playback instead.")
    parser.add_argument("--force_restart", action='store_true',
                        help="Whether force to restart the roscore.")
    parser.add_argument("--vis_calib", action='store_true', default=False, \
                        help="Visualize the calibration process.")
    parser.add_argument("--display", default="001000", \
                        help="0/000000: No display;"
                             "1/000001: source input;"
                             "2/000010: hand;"
                             "4/000100: robot;"
                             "8/001000: puzzle;"
                             "16/010000: postprocessing;"
                             "32/100000: puzzle board;"
                             "You can use decimal or binary as the input.")
    parser.add_argument("--survelliance_system", action='store_true',
                        help="Whether to apply survelliance_system.")
    parser.add_argument("--puzzle_solver", action='store_true',
                        help="Whether to apply puzzle_solver.")
    parser.add_argument("--puzzle_solver_mode", default=0,
                        help="0: Set the first frame as the solution img; (For Test_human_activity/Test_puzzle_progress/Test_system_general)"
                             "1: Calibration based on a rosbag recording; (For Test_calibration)"
                             "2: Run on the rosbag recording assuming the calibration board is already saved. (For Test_puzzle_solving)")
    parser.add_argument("--puzzle_solver_SolBoard", default='caliSolBoard.obj',
                        help="The saving path to a .obj instance")
    parser.add_argument("--state_analysis", action='store_true',
                        help="Whether to apply the state analysis (move/not move/no hand). Display is automatically enabled.")
    parser.add_argument("--activity_interpretation", action='store_true',
                        help="Whether to interpret the human's activity (piece status change & human activity change (inferred from piece status change)),"
                             "Display is automatically enabled.")
    parser.add_argument("--verbose", action='store_true',
                        help="Whether to debug the system.")
    parser.add_argument("--save_to_file", action='store_true',
                        help="Whether save to files, the default file location is the same as the rosbag or realtime.")
    parser.add_argument("--debug_individual_folder", action='store_true',
                        help="Whether save files into different folders. More convenient for debug.")
    args = parser.parse_args()

    cfg = CfgNode_SurvRunner()
    cfg.load_defaults()

    if args.yfile is not None:
        cfg.merge_from_files([args.yfile])

    return args, cfg


class ImageListener:
    def __init__(self, opt, cfg):

        self.opt = opt
        self.cfg = cfg

        # Convert display code
        self.opt.display = display_option_convert(self.opt.display)

        # Initialize the saving folder

        if self.opt.save_to_file:
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

        if self.opt.puzzle_solver_mode == 1:
            mea_mode = 'sol'
        else:
            mea_mode = 'test'

        # Build up the surveillance system from the rosbag
        configs_surv = bParams(
            visualize=False,
            vis_calib=self.opt.vis_calib,
            ros_pub=False,
            # Postprocessing
            bound_limit=[200, 200, 50, 50],
            # @< The ignored region area. Top/Bottom/Left/Right. E.g., Top: 200, 0-200 is ignored.
            mea_mode=mea_mode,  # @< The mode for the postprocessing function, 'test' or 'sol'.
            mea_test_r=150,  # @< The circle size in the postprocessing for the measured board.
            # mea_test_r=100,  # @< The circle size in the postprocessing for the measured board.
            mea_sol_r=200,  # @< The circle size in the postprocessing for the solution board.
            hand_radius=200  # @< The hand radius set by the user.
        )

        # build the surveillance deployer
        self.surv = BaseSurveillanceDeploy.buildFromRosbag(rosbag_file, configs_surv, self.cfg)

        # Build up the puzzle solver
        configs_puzzleSolver = ParamRunner(
            areaThresholdLower=1000,  # @< The area threshold (lower) for the individual puzzle piece.
            areaThresholdUpper=8000,  # @< The area threshold (upper) for the individual puzzle piece.
            pieceConstructor=Template,
            lengthThresholdLower=1000,
            BoudingboxThresh=(20, 100),  # @< The bounding box threshold for the size of the individual puzzle piece.
            tauDist=100,  # @< The radius distance determining if one piece is at the right position.
            hand_radius=200,  # @< The radius distance to the hand center determining the near-by pieces.
            tracking_life_thresh=15,
            # @< Tracking life for the pieces, it should be set according to the processing speed.
            # solution_area=[600,800,400,650], # @< The solution area, [xmin, xmax, ymin, ymax]. We will perform frame difference in this area to locate the touching pieces.
            # It is set by the calibration result of the solution board.
        )
        self.puzzleSolver = RealSolver(configs_puzzleSolver)

        # State analysis
        self.state_parser = StateEstimator(
            signal_number=1,
            signal_names=["location"],
            state_number=1,
            state_names=["Move"],
            # move_th=50, # @< The threshold for determining the moving status. Note that this thresh only works well with low sampling rate.
            move_th=25,
        )

        # Initialize a subscriber
        Images_sub([test_rgb_topic, test_dep_topic], callback_np=self.callback_rgbd)

        # Initialize the activity label subscriber, decoder, and the label storage if needed
        self.activity_label = None
        if args.read_activity:
            rospy.Subscriber(test_activity_topic, UInt8, callback=self.callback_activity, queue_size=1)
            self.act_decoder = ActDecoder()

        # Activity analysis related
        # Initialized with NoHand
        self.move_state_history = None

        # Fig for puzzle piece status display
        self.status_window = None
        self.activity_window = None

        # ROS support
        self.postImg_pub = Image_pub(topic_name=postImg_topic)
        self.visibleMask_pub = Image_pub(topic_name=visibleMask_topic)
        self.hTracker_BEV_pub = rospy.Publisher(hTracker_BEV_topic, String, queue_size=5)

        String_sub(puzzle_solver_info_topic, String, callback_np=self.callback_puzzle_solver_info) # Not important for now
        String_sub(status_history_topic, String, callback_np=self.callback_status_history)
        String_sub(loc_history_topic, String, callback_np=self.callback_loc_history)

        # Data captured
        self.puzzle_solver_info = None
        self.status_history = None
        self.loc_history = None

        print("Initialization ready, waiting for the data...")

    def callback_puzzle_solver_info(self, msg):

        puzzle_solver_info = convert_ROS2dict(msg)

        with lock:
            self.puzzle_solver_info = puzzle_solver_info

    def callback_status_history(self, msg):

        status_history = convert_ROS2dict(msg)

        # We find that during the encoding & decoding, the key may change from int to str.
        # We need to convert the key back to int.
        # Todo: Maybe there is a better way to do this.
        # In the end, k: [PieceStatus(Enum), PieceStatus(Enum), ...]
        if len(status_history.keys()) > 0:
            status_history_processed = {}
            for key in status_history.keys():
                if isinstance(key, str):
                    status_history_processed[int(key)] = [PieceStatus(x) for x in status_history[key]]
                else:
                    status_history_processed[key] = [PieceStatus(x) for x in status_history[key]]

            status_history = status_history_processed

        with lock:
            self.status_history = status_history

    def callback_loc_history(self, msg):

        loc_history = convert_ROS2dict(msg)

        # We find that during the encoding & decoding, the key may change from int to str.
        # We need to convert the key back to int.
        # Todo: Maybe there is a better way to do this.
        # In the end, k: [array([x1, y1]), array([x2, y2]), ...]
        if len(loc_history.keys()) > 0:
            loc_history_processed = {}
            for key in loc_history.keys():
                if isinstance(key, str):
                    loc_history_processed[int(key)] = [np.array(x) for x in loc_history[key]]
                else:
                    loc_history_processed[key] = [np.array(x) for x in loc_history[key]]

            loc_history = loc_history_processed

        with lock:
            self.loc_history = loc_history

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

            global call_back_id

            if self.RGB_np is None:
                if self.opt.verbose:
                    print('No data')
                return

            rgb_frame_stamp = copy.deepcopy(self.rgb_frame_stamp)

            # Skip images with the same timestamp as the previous one
            if rgb_frame_stamp != None and self.rgb_frame_stamp_prev == rgb_frame_stamp:

                time.sleep(0.001)
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

            if self.opt.save_to_file:
                # Save for debug
                cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_rgb.png'),
                            self.RGB_np[:, :, ::-1])

            if self.opt.survelliance_system:
                self.surv.process(RGB_np, D_np)

                # For demo
                humanImg = self.surv.humanImg
                robotImg = self.surv.robotImg
                puzzleImg = self.surv.puzzleImg  # @< Directly from surveillance system (without postprocessing)
                humanMask = self.surv.humanMask

                # For further processing
                postImg = self.surv.meaBoardImg
                visibleMask = self.surv.visibleMask
                hTracker = self.surv.hTracker
                hTracker_BEV = self.surv.scene_interpreter.get_trackers("human", BEV_rectify=True)  # (2, 1)

                if self.opt.save_to_file:
                    # Save for debug

                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_hand.png'),
                                humanImg[:, :, ::-1])
                    # cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_handMask.png'),
                    #             humanMask)
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_puzzle.png'),
                                postImg[:, :, ::-1])
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_visibleMask.png'),
                                visibleMask)

                    with open(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_hTracker.npy'),
                              'wb') as f:
                        np.save(f, hTracker_BEV)

            # Display
            if self.opt.display[0]:
                if self.activity_label is not None:
                    RGB_np_withLabel = cv2.putText(np.float32(RGB_np.copy()), self.activity_label, (10, 60),
                                                   cv2.FONT_HERSHEY_SIMPLEX,
                                                   2.0, (255, 0, 0), 5)
                    RGB_np_withLabel = np.uint8(RGB_np_withLabel)
                    display_images_cv([RGB_np_withLabel[:, :, ::-1]], ratio=0.5, window_name="Source RGB")
                else:
                    display_images_cv([RGB_np[:, :, ::-1]], ratio=0.5, window_name="Source RGB")
            if self.opt.display[1]:
                display_images_cv([humanImg[:, :, ::-1]], ratio=0.5, window_name="Hand layer")
            if self.opt.display[2]:
                display_images_cv([robotImg[:, :, ::-1]], ratio=0.5, window_name="Robot layer")
            if self.opt.display[3]:
                display_images_cv([puzzleImg[:, :, ::-1]], ratio=0.5, window_name="Puzzle layer")
            if self.opt.display[4]:
                display_images_cv([postImg[:, :, ::-1]], ratio=0.5,
                                  window_name="Postprocessing (Input to the puzzle solver)")


            ########################################################

            # If there is at least one display command
            if any(self.opt.display):
                cv2.waitKey(1)

            if self.opt.state_analysis:
                # Hand moving states
                # Todo: here I only invoke the state parser when the hand is detected (hTracker is not None)
                # Todo: Should be moved to state parser in the future
                # This might cause non-synchronization with the puzzle states.
                # So might need to set the self.move_state to indicator value when the hand is not detected.
                if hTracker is None:
                    # -1: NoHand; 0: NoMove; 1: Move
                    self.move_state = -1

                    stateImg = cv2.putText(RGB_np.copy(), "No Hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                                           [255, 0, 0], 5)

                else:
                    # Get the tracker
                    self.state_parser.process([hTracker])
                    stateImg = self.state_parser.plot_states(RGB_np.copy())

                    # NOTE: The moving state is obtained here.
                    # The return is supposed to be of the shape (N_state, ), where N_state is the number of states,
                    # since it was designed to include extraction of all the states.
                    # Since the puzzle states is implemented elsewhere, the N_state is 1, hence index [0]
                    self.move_state = self.state_parser.get_states()[0]

                display_images_cv([stateImg[:, :, ::-1]], ratio=0.5, window_name="Move States")

                if self.opt.save_to_file:
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_state.png'),
                                stateImg[:, :, ::-1])

                print(f'Hand state: {self.move_state}')


            # ROS support
            self.postImg_pub.pub(postImg)
            self.visibleMask_pub.pub(visibleMask)

            info_dict = {
                'hTracker_BEV': hTracker_BEV if hTracker_BEV is None else hTracker_BEV.tolist(),
            }

            self.hTracker_BEV_pub.publish(convert_dict2ROS(info_dict))

            ########################################################
            # The following is processed inside the puzzle solver ROS (with fewer debug support)


            # # We need (postImg, visibleMask, hTracker_BEV) from the the surveillance system
            # # The main system will get (the solution board size, plan, progress, bMeasImage, bTrackImage_SolID, bSolImage) from the puzzle solver
            # if self.opt.puzzle_solver:
            #     # Work on the puzzle pieces
            #
            #     if self.opt.puzzle_solver_mode == 0:
            #         if call_back_id == 0:
            #             # Initialize the SolBoard using the very first frame.
            #             self.puzzleSolver.setSolBoard(postImg)
            #
            #             print(
            #                 f'Number of puzzle pieces registered in the solution board: {self.puzzleSolver.theManager.solution.size()}')
            #
            #             if self.opt.activity_interpretation:
            #                 self.status_window = DynamicDisplay(
            #                     ParamDynamicDisplay(num=self.puzzleSolver.theManager.solution.size(),
            #                                         window_title='Status Change'))
            #                 self.activity_window = DynamicDisplay(
            #                     ParamDynamicDisplay(num=self.puzzleSolver.theManager.solution.size(),
            #                                         status_label=['NONE', 'MOVE'], ylimit=1,
            #                                         window_title='Activity Change'))
            #
            #             # # Debug only
            #             if self.opt.verbose:
            #                 cv2.imshow('debug_source', RGB_np[:, :, ::-1])
            #                 cv2.imshow('debug_humanMask', humanMask)
            #                 cv2.imshow('debug_puzzleImg', puzzleImg[:, :, ::-1])
            #                 cv2.imshow('debug_postImg', postImg[:, :, ::-1])
            #                 cv2.imshow('debug_solBoard',
            #                            self.puzzleSolver.theManager.solution.toImage(ID_DISPLAY=True)[:, :, ::-1])
            #                 cv2.waitKey()
            #         # Plan not used yet
            #         plan = self.puzzleSolver.process(postImg, visibleMask, hTracker_BEV)
            #
            #     elif self.opt.puzzle_solver_mode == 1:
            #         # Calibration process
            #         # Plan not used yet
            #         plan = self.puzzleSolver.calibrate(postImg, visibleMask, hTracker_BEV)
            #     elif self.opt.puzzle_solver_mode == 2:
            #
            #         # Initialize the SolBoard with saved board at the very first frame.
            #         if call_back_id == 0:
            #             self.puzzleSolver.setSolBoard(postImg, self.opt.puzzle_solver_SolBoard)
            #
            #             print(
            #                 f'Number of puzzle pieces registered in the solution board: {self.puzzleSolver.theManager.solution.size()}')
            #
            #             if self.opt.activity_interpretation:
            #                 self.status_window = DynamicDisplay(
            #                     ParamDynamicDisplay(num=self.puzzleSolver.theManager.solution.size(),
            #                                         window_title='Status Change'))
            #                 self.activity_window = DynamicDisplay(
            #                     ParamDynamicDisplay(num=self.puzzleSolver.theManager.solution.size(),
            #                                         status_label=['NONE', 'MOVE'], ylimit=1,
            #                                         window_title='Activity Change'))
            #
            #             # # Debug only
            #             if self.opt.verbose:
            #                 cv2.imshow('debug_source', RGB_np[:, :, ::-1])
            #                 cv2.imshow('debug_humanMask', humanMask)
            #                 cv2.imshow('debug_puzzleImg', puzzleImg[:, :, ::-1])
            #                 cv2.imshow('debug_postImg', postImg[:, :, ::-1])
            #                 cv2.imshow('debug_solBoard',
            #                            self.puzzleSolver.theManager.solution.toImage(ID_DISPLAY=True)[:, :, ::-1])
            #                 cv2.waitKey()
            #
            #         # Plan not used yet
            #         plan = self.puzzleSolver.process(postImg, visibleMask, hTracker_BEV, run_solver=False)
            #         cv2.waitKey(1)
            #     else:
            #         raise RuntimeError('Wrong puzzle_solver_mode!')
            #
            #     if self.opt.display[5]:
            #         # Display measured/tracked/solution board
            #         # display_images_cv([self.puzzleSolver.bMeasImage[:, :, ::-1], self.puzzleSolver.bTrackImage[:, :, ::-1], self.puzzleSolver.bSolImage[:, :, ::-1]],
            #         #                   ratio=0.5, window_name="Measured/Tracking/Solution board")
            #
            #         # Display measured/tracked(ID from solution board)/solution board
            #         display_images_cv(
            #             [self.puzzleSolver.bMeasImage[:, :, ::-1], self.puzzleSolver.bTrackImage_SolID[:, :, ::-1],
            #              self.puzzleSolver.bSolImage[:, :, ::-1]],
            #             ratio=0.5, window_name="Measured/Tracked/Solution board")
            #
            #         cv2.waitKey(1)
            #
            #     if self.opt.puzzle_solver_mode != 1 and self.opt.save_to_file:
            #         # Save for debug
            #         cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_bMeas.png'),
            #                     self.puzzleSolver.bMeasImage[:, :, ::-1])
            #         cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_bTrack_SolID.png'),
            #                     self.puzzleSolver.bTrackImage_SolID[:, :, ::-1])
            #
            #     # Compute progress
            #     # Note that the solution board should be correct, otherwise it will fail.
            #     if self.opt.puzzle_solver_mode != 1:
            #         try:
            #             thePercent = self.puzzleSolver.progress(USE_MEASURED=False)
            #             print(f"Progress: {thePercent}")
            #         except:
            #             print('Double check the solution board to make it right.')

            ########################################################

            # Todo: Ideally, we should have a separated node to handle the following task
            if self.opt.activity_interpretation:

                if self.puzzle_solver_info is not None and self.status_history is not None and self.loc_history is not None:

                    if self.status_window is None and self.activity_window is None:
                        self.status_window = DynamicDisplay(
                            ParamDynamicDisplay(num=self.puzzle_solver_info['solution_board_size'],
                                                window_title='Status Change'))
                        self.activity_window = DynamicDisplay(
                            ParamDynamicDisplay(num=self.puzzle_solver_info['solution_board_size'],
                                                status_label=['NONE', 'MOVE'], ylimit=1,
                                                window_title='Activity Change'))

                    status_data = np.zeros(len(self.status_history))
                    activity_data = np.zeros(len(self.status_history))

                    for i in range(len(status_data)):
                        try:
                            status_data[i] = self.status_history[i][-1].value
                        except:
                            status_data[i] = PieceStatus.UNKNOWN.value

                        # Debug only
                        # if len(self.status_history[i])>=2 and \
                        #         np.linalg.norm(self.loc_history[i][-1] - self.loc_history[i][-2]) > 10:
                        #     print('!')

                        if len(self.status_history[i]) >= 2 and \
                                self.status_history[i][-1] == PieceStatus.MEASURED and \
                                self.status_history[i][-2] != PieceStatus.MEASURED and \
                                np.linalg.norm(self.loc_history[i][-1] -
                                               self.loc_history[i][-2]) > 30:
                            activity_data[i] = 1
                            print('Move activity detected.')

                        else:
                            activity_data[i] = 0

                    self.status_window((call_back_id, status_data))
                    self.activity_window((call_back_id, activity_data))
            ########################################################

            # # The activity_interpretation module will get (status_history, loc_history) from the puzzle solver
            # if self.opt.activity_interpretation:
            #
            #     # TODO: Need to be moved to somewhere else
            #     status_data = np.zeros(len(self.puzzleSolver.thePlanner.status_history))
            #     activity_data = np.zeros(len(self.puzzleSolver.thePlanner.status_history))
            #
            #     for i in range(len(status_data)):
            #         try:
            #             status_data[i] = self.puzzleSolver.thePlanner.status_history[i][-1].value
            #         except:
            #             status_data[i] = PieceStatus.UNKNOWN.value
            #
            #         # Debug only
            #         # if len(self.puzzleSolver.thePlanner.status_history[i])>=2 and \
            #         #         np.linalg.norm(self.puzzleSolver.thePlanner.loc_history[i][-1] - self.puzzleSolver.thePlanner.loc_history[i][-2]) > 10:
            #         #     print('!')
            #
            #         if len(self.puzzleSolver.thePlanner.status_history[i]) >= 2 and \
            #                 self.puzzleSolver.thePlanner.status_history[i][-1] == PieceStatus.MEASURED and \
            #                 self.puzzleSolver.thePlanner.status_history[i][-2] != PieceStatus.MEASURED and \
            #                 np.linalg.norm(self.puzzleSolver.thePlanner.loc_history[i][-1] -
            #                                self.puzzleSolver.thePlanner.loc_history[i][-2]) > 30:
            #             activity_data[i] = 1
            #             print('Move activity detected.')
            #
            #         else:
            #             activity_data[i] = 0
            #
            #     self.status_window((call_back_id, status_data))
            #     self.activity_window((call_back_id, activity_data))

            print(f"The processed test frame id: {call_back_id} ")
            call_back_id += 1

        # Only applied when working on rosbag playback
        if self.opt.real_time is False:

            global timestamp_beginning
            global timestamp_ending
            global roscore_proc

            print(f'Current frame time: {np.round(rgb_frame_stamp - timestamp_beginning, 2)}s')
            print('\n\n')

            # # Debug only
            if self.opt.verbose:
                print(f'Last frame time: {np.round(timestamp_ending - timestamp_beginning, 2)}s')

            # We ignore the last 2 seconds
            if timestamp_ending is not None and abs(rgb_frame_stamp - timestamp_ending) < 2:

                # if self.opt.puzzle_solver_mode == 1:
                #     # Only for calibration process
                #     if self.puzzleSolver.theCalibrated.size() > 0:
                #         # Save for future usage
                #         with open(self.opt.puzzle_solver_SolBoard, 'wb') as fp:
                #             pickle.dump(self.puzzleSolver.theCalibrated, fp)
                #
                #         print(
                #             f'Number of puzzle pieces registered in the solution board: {self.puzzleSolver.theCalibrated.size()}')
                #         print(f'Bounding box of the solution area: {self.puzzleSolver.theCalibrated.boundingBox()}')
                #         cv2.imshow('debug_solBoard',
                #                    self.puzzleSolver.theCalibrated.toImage(ID_DISPLAY=True)[:, :, ::-1])
                #         cv2.waitKey()
                #     else:
                #         print('No piece detected.')

                print('Shut down the system.')
                rospy.signal_shutdown('Finished')
                # Stop the roscore if started from the script
                if roscore_proc is not None:
                    terminate_process_and_children(roscore_proc)
        else:
            print('\n\n')


if __name__ == "__main__":

    # Parse arguments
    args, cfg = get_args()

    ##################################
    # Local configuration for debug

    # # General setting
    # args.save_to_file = True
    # args.debug_individual_folder = True
    # args.verbose = True
    args.force_restart = True

    # # # For more modules setting
    # args.survelliance_system = True
    # args.puzzle_solver = True
    # args.state_analysis = True
    # args.activity_interpretation = True
    # args.puzzle_solver_mode = 0

    # # Display setting
    # "0/000000: No display;"
    # "1/000001: source input;"
    # "2/000010: hand;"
    # "4/000100: robot;"
    # "8/001000: puzzle;"
    # "16/010000: postprocessing;"
    # "32/100000: puzzle board;"
    # "You can use decimal or binary as the input."

    # args.display = '001011' # @< For most debug purposes on surveillance system
    # args.display = '110001' # @< For most debug purposes on puzzle solver
    # args.display = '000001' # @< For most debug purposes on activity analysis

    ##################################

    # # Option 0: Test puzzle solver
    # # args.rosbag_name = 'data/Testing/Yunzhi/Test_human_activity/activity_multi_free_8.bag'
    # args.rosbag_name = 'data/Testing/Yunzhi/Test_system_general/debug_system_1.bag'
    # args.survelliance_system = True
    # args.puzzle_solver = True
    # args.state_analysis = True
    # args.activity_interpretation = True
    # args.puzzle_solver_mode = 0
    # args.display = '110001'

    # # Option 1: Calibration
    # args.rosbag_name = 'data/Testing/Yunzhi/Test_puzzle_solving/tangled_1_sol.bag'
    # args.survelliance_system = True
    # args.puzzle_solver = True
    # args.puzzle_solver_mode = 1
    # args.display = '010001'

    # # Option 2: Test puzzle solver with solution board set up (option 1 must be run in advance to get the solution board)
    # args.rosbag_name = 'data/Testing/Yunzhi/Test_puzzle_solving/tangled_1_work.bag'
    # args.survelliance_system = True
    # args.puzzle_solver = True
    # args.state_analysis = True
    # args.activity_interpretation = True
    # args.puzzle_solver_mode = 2
    # args.display = '111001'

    # A special case for option 2
    args.rosbag_name = 'testing/data/tangled_1_work.bag'
    args.survelliance_system = True
    args.puzzle_solver = True
    args.activity_interpretation = True
    args.puzzle_solver_mode = 2
    args.display = 1
    args.force_restart = False # Do not force restart, otherwise the other modules will be killed

    ###################################

    # update the args about the existence of the activity topic
    rosbag_file = os.path.join(args.fDir, args.rosbag_name)
    print(rosbag_file)
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

    listener = ImageListener(args, cfg)

    plt.ion()
    if args.real_time is False:
        # Get basic info from the rosbag
        info_dict = yaml.safe_load(
            subprocess.Popen(['rosbag', 'info', '--yaml', rosbag_file], stdout=subprocess.PIPE).communicate()[0])

        timestamp_beginning = info_dict['start']
        timestamp_ending = info_dict['end']

        print('Playback the rosbag recordings')

        # Need to start later for initialization
        # May need to slow down the publication otherwise the subscriber won't be able to catch it
        # -d:delay; -r:rate; -s:skip; -q no console display
        command = "rosbag play {} -d 2 -r 1 -s 15 -q --topic {} {} {}".format(
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

    if args.save_to_file and args.debug_individual_folder:
        # Mainly for debug
        def resave_to_folder(target):
            file_list = glob.glob(os.path.join(listener.opt.save_folder, f'*{target}.png'))

            if os.path.exists(f'{target}'):
                shutil.rmtree(f'{target}')
            os.makedirs(f'{target}', exist_ok=True)

            for file_path in file_list:
                shutil.copyfile(file_path, os.path.join(f'{target}', os.path.basename(file_path)))


        target_list = ['bTrack_SolID']

        for i in target_list:
            resave_to_folder(i)