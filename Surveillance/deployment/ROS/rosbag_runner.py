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
from Surveillance.deployment.utils import display_option_convert, calc_closest_factors

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
from Surveillance.activity.FSM import Pick, Place
from Surveillance.activity.utils import DynamicDisplay, ParamDynamicDisplay

# configs
test_rgb_topic = "/test_rgb"
test_dep_topic = "/test_depth"
test_activity_topic = "/test_activity"

# preparation
lock = threading.Lock()

timestamp_beginning = None
timestamp_ending = None
roscore_proc = None

# To be built
call_back_id = 0

def get_args():
    parser = argparse.ArgumentParser(description="Surveillance runner on the pre-saved rosbag file")
    parser.add_argument("--fDir", type=str, default="./", \
                        help="The folder's name.")
    parser.add_argument("--rosbag_name", type=str, default="data/Testing/Yunzhi/Test_human_activity/activity_multi_free_3.bag", \
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
    parser.add_argument("--survelliance_system", action='store_true', \
                        help="Whether to apply survelliance_system.")
    parser.add_argument("--puzzle_solver", action='store_true', \
                        help="Whether to apply puzzle_solver.")
    parser.add_argument("--state_analysis", action='store_true', \
                        help="Whether to apply the state analysis. Display is automatically enabled.")
    parser.add_argument("--activity_interpretation", action='store_true', \
                        help="Whether to interpret the human's activity. Display is automatically enabled.")
    # parser.add_argument("--near_hand_demo", action='store_true', \
    #                     help="Whether to visualize near hand pieces.")
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

        if self.opt.save_to_file:
            if not self.opt.real_time:
                self.opt.save_folder = Path(self.opt.rosbag_name).stem
            else:
                self.opt.save_folder = 'realtime'

            # Clear up the space
            if os.path.exists(self.opt.save_folder):
                shutil.rmtree(self.opt.save_folder)
            os.makedirs(self.opt.save_folder, exist_ok=True)

            if os.path.exists('activity'):
                shutil.rmtree('activity')
            os.makedirs('activity', exist_ok=True)

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
            bound_limit = [200,200,50,50], # @< The ignored region area. Top/Bottom/Left/Right
            mea_test_r = 100,  # @< The circle size in the postprocessing for the measured board.
            mea_sol_r = 300,  # @< The circle size in the postprocessing for the solution board.
            hand_radius = 200  # @< The hand radius set by the user.
        )

        # build the surveillance deployer
        self.surv = BaseSurveillanceDeploy.buildFromRosbag(rosbag_file, configs_surv)

        # Build up the puzzle solver
        configs_puzzleSolver = ParamRunner(
            areaThresholdLower=2000,
            areaThresholdUpper=8000,
            pieceConstructor=Template,
            lengthThresholdLower=1000,
            areaThresh=1000,
            BoudingboxThresh=(20, 100),
            tauDist=100, # @< The radius distance determining if one piece is at the right position.
            hand_radius=200, # @< The radius distance to the hand center determining the near-by pieces.
            tracking_life_thresh=15 # @< Tracking life for the pieces, it should be set according to the processing speed.
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

        self.pick_model = Pick()
        self.place_model = Place()

        # Initialize fig for puzzle piece status display
        if self.opt.activity_interpretation:
            self.status_window = None
            self.activity_window = None

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
                cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_rgb.png'), self.RGB_np[:, :, ::-1])

            if self.opt.survelliance_system:
                self.surv.process(RGB_np, D_np)

                # For demo
                humanImg = self.surv.humanImg
                robotImg = self.surv.robotImg
                puzzleImg = self.surv.puzzleImg # @< Directly from surveillance system (without postprocessing)
                humanMask = self.surv.humanMask
                # nearHandImg = self.surv.humanAndhumanImg

                # For further processing
                postImg = self.surv.meaBoardImg
                visibleMask = self.surv.visibleMask
                # postMask = self.surv.meaBoardMask
                hTracker = self.surv.hTracker
                hTracker_BEV = self.surv.scene_interpreter.get_trackers("human", BEV_rectify=True)  # (2, 1)

                # Note: It seems that this process is unnecessary to us as we have integrated the nearHand into pick & place interpretation
                # For near-human-hand puzzle pieces.
                # @note there may be false positives
                # pTracker_BEV = self.surv.scene_interpreter.get_trackers("puzzle", BEV_rectify=True)  # (2, N)
                # near_human_puzzle_idx = self.surv.near_human_puzzle_idx # @< pTracker_BEV is the trackpointers of all the pieces.
                # print('Idx from puzzle solver:', near_human_puzzle_idx) # @< The index of the pTracker_BEV that is near the human hand

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

                    with open(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_hTracker.npy'), 'wb') as f:
                        np.save(f, hTracker_BEV)
                    # cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_nearHand.png'),
                    #             nearHandImg[:, :, ::-1])

            # Display
            if self.opt.display[0]:
                if self.activity_label is not None:
                    RGB_np_withLabel = cv2.putText(np.float32(RGB_np.copy()), self.activity_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
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
                display_images_cv([postImg[:, :, ::-1]], ratio=0.5, window_name="Postprocessing (Input to the puzzle solver)")
            # if self.opt.near_hand_demo:
            #     self.surv.vis_near_hand_puzzles()
            #     display_images_cv([nearHandImg[:, :, ::-1]], ratio=0.5, window_name="nearHandImg")

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

                    stateImg = cv2.putText(RGB_np.copy(), "No Hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, [255, 0, 0], 5)

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

                # if self.opt.save_to_file:
                #     cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_state.png'), stateImg[:, :, ::-1])

                print(f'Hand state: {self.move_state}')

            if self.opt.puzzle_solver:
                # Work on the puzzle pieces

                # Todo: Currently, initialize the SolBoard with the very first frame.
                # We assume SolBoard is perfect (all the pieces have been recognized successfully)
                # We can hack it with something outside
                if call_back_id == 0:

                    self.puzzleSolver.setSolBoard(postImg)

                    print(f'Number of puzzle pieces registered in the solution board: {self.puzzleSolver.theManager.solution.size()}')

                    if self.opt.activity_interpretation:
                        self.status_window = DynamicDisplay(ParamDynamicDisplay(num=self.puzzleSolver.theManager.solution.size(), window_title='Status Change'))
                        self.activity_window = DynamicDisplay(ParamDynamicDisplay(num=self.puzzleSolver.theManager.solution.size(), status_label=['NONE', 'MOVE'], ylimit=1, window_title='Activity Change'))

                    # # Debug only
                    if self.opt.verbose:
                        cv2.imshow('debug_source', RGB_np[:, :, ::-1])
                        cv2.imshow('debug_humanMask', humanMask)
                        cv2.imshow('debug_puzzleImg', puzzleImg[:, :, ::-1])
                        cv2.imshow('debug_postImg', postImg[:, :, ::-1])
                        cv2.imshow('debug_solBoard', self.puzzleSolver.theManager.solution.toImage()[:, :, ::-1])
                        cv2.waitKey()

                # Plan not used yet
                plan, id_dict, hand_activity = self.puzzleSolver.process(postImg, visibleMask, hTracker_BEV)

                # # Note: It seems that this process is unnecessary to us as we have integrated the nearHand into pick & place interpretation
                # # @note there may be false negatives
                # print('ID from puzzle solver:', id_dict)
                print('Hand activity:', hand_activity)

                if self.opt.display[5]:
                    # Display measured/tracked/solution board
                    # display_images_cv([self.puzzleSolver.bMeasImage[:, :, ::-1], self.puzzleSolver.bTrackImage[:, :, ::-1], self.puzzleSolver.bSolImage[:, :, ::-1]],
                    #                   ratio=0.5, window_name="Measured/Tracking/Solution board")

                    # Display the original measured/tracked(ID from solution board)/solution board
                    display_images_cv(
                        [self.puzzleSolver.bMeasImage[:, :, ::-1], self.puzzleSolver.bTrackImage_SolID[:, :, ::-1],
                         self.puzzleSolver.bSolImage[:, :, ::-1]],
                        ratio=0.5, window_name="Measured/Tracked/Solution board")

                    cv2.waitKey(1)

                if self.opt.save_to_file:
                    # Save for debug
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_bMeas.png'), self.puzzleSolver.bMeasImage[:, :, ::-1])
                    cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_bTrack.png'), self.puzzleSolver.bTrackImage[:, :, ::-1])

                # Compute progress
                # Note that the solution board should be correct, otherwise it will fail.
                try:
                    thePercent = self.puzzleSolver.progress()
                    print(f"Progress: {thePercent}")
                except:
                    print('Double check the solution board to make it right.')

            if self.opt.activity_interpretation:

                # Todo: Need to be moved to somewhere else
                status_data = np.zeros(len(self.puzzleSolver.thePlanner.status_history))
                activity_data = np.zeros(len(self.puzzleSolver.thePlanner.status_history))

                for i in range(len(status_data)):
                    status_data[i] = self.puzzleSolver.thePlanner.status_history[i][-1].value

                    # Debug only
                    # if len(self.puzzleSolver.thePlanner.status_history[i])>=2 and \
                    #         np.linalg.norm(self.puzzleSolver.thePlanner.loc_history[i][-1] - self.puzzleSolver.thePlanner.loc_history[i][-2]) > 10:
                    #     print('!')

                    if len(self.puzzleSolver.thePlanner.status_history[i])>=2 and \
                        self.puzzleSolver.thePlanner.status_history[i][-1] == PieceStatus.MEASURED and \
                            self.puzzleSolver.thePlanner.status_history[i][-2] != PieceStatus.MEASURED and \
                                np.linalg.norm(self.puzzleSolver.thePlanner.loc_history[i][-1] - self.puzzleSolver.thePlanner.loc_history[i][-2]) > 10:
                            activity_data[i]= 1
                            print('Move activity detected.')

                            if i==0:
                                print('!!')
                    else:
                        activity_data[i]= 0

                self.status_window((call_back_id, status_data))
                self.activity_window((call_back_id, activity_data))
                # plt.show()


            # # Todo: Need to be moved to somewhere else
            #
            # if self.opt.activity_interpretation:

                # # Todo: To be moved to state
                # # Hack for NoHand move_state
                # # Eventually we only have two states for FSM: 0 or 1
                # if self.move_state_history is not None and self.move_state_history == -1 and self.move_state >= 0:
                #     self.move_state_final = 1
                # elif self.move_state_history is not None and self.move_state_history >= 0 and self.move_state == -1:
                #     self.move_state_final = 1
                # elif self.move_state == -1:
                #     self.move_state_final = 0
                # else:
                #     self.move_state_final = self.move_state
                #
                # print(f'Hand state history: {self.move_state_history }')
                #
                # # Pick
                # # Check if hand is invisible for a while or we have recognized the pick action
                # if (self.move_state_history == -1 and self.move_state == -1) or self.pick_model.state == 'E':
                #     self.pick_model.reset()
                # elif self.pick_model.state != 'D':
                #     if self.move_state_final == 1:
                #         self.pick_model.move()
                #     else:
                #         self.pick_model.stop()
                # if self.pick_model.state == 'D':
                #     if hand_activity == 1:
                #         self.pick_model.piece_disappear()
                #     else:
                #         self.pick_model.no_piece_disappear()
                #
                # # Place
                # if (self.move_state_history == -1 and self.move_state == -1) or self.place_model.state == 'E':
                #     self.place_model.reset()
                # elif self.place_model.state != 'D':
                #     if self.move_state_final == 1:
                #         self.place_model.move()
                #     else:
                #         self.place_model.stop()
                # if self.place_model.state == 'D':
                #     if hand_activity == 2:
                #         self.place_model.piece_added()
                #     else:
                #         self.place_model.no_piece_added()
                #
                # # Debug only
                # print(f'Pick model state: {self.pick_model.state}')
                # print(f'Place model state: {self.place_model.state}')
                #
                # # Todo: Adhoc display
                # # The reason why we do not use FSM for place efficiently is that user may not strictly follow our protocol.
                # # They may move the piece too fast.
                # activityImg = RGB_np.copy()
                #
                # # if self.pick_model.state == 'E':
                # if hand_activity == 1:
                #     activityImg = cv2.putText(np.float32(activityImg), 'PICK', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                #                          2.0, (255, 0, 0), 5)
                #     activityImg = np.uint8(activityImg)
                # if hand_activity == 2:
                #     activityImg = cv2.putText(np.float32(activityImg), 'PLACE', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                #                          2.0, (255, 0,   0), 5)
                #     activityImg = np.uint8(activityImg)
                # display_images_cv([activityImg[:, :, ::-1]], ratio=0.5, window_name="Activity")
                #
                # if self.opt.save_to_file:
                #     # Save for debug
                #     cv2.imwrite(os.path.join(self.opt.save_folder, f'{str(call_back_id).zfill(4)}_activity.png'), activityImg[:, :, ::-1])
                #     cv2.imwrite(os.path.join('activity', f'{str(call_back_id).zfill(4)}_activity.png'), activityImg[:, :, ::-1])
                #
                # self.move_state_history = self.move_state

            print(f"The processed test frame id: {call_back_id} ")
            call_back_id += 1

        # Only applied when working on rosbag playback
        if self.opt.real_time is False:

            global timestamp_beginning
            global timestamp_ending
            global roscore_proc

            print(f'Current frame time: {np.round(rgb_frame_stamp-timestamp_beginning,2)}s')
            print('\n\n')

            # # Debug only
            if self.opt.verbose:
                print(f'Last frame time: {np.round(timestamp_ending-timestamp_beginning,2)}s')

            # We ignore the last 2 seconds
            if timestamp_ending is not None and abs(rgb_frame_stamp - timestamp_ending) < 2:
                print('Shut down the system.')
                rospy.signal_shutdown('Finished')
                # Stop the roscore if started from the script
                if roscore_proc is not None:
                    terminate_process_and_children(roscore_proc)
        else:
            print('\n\n')
if __name__ == "__main__":

    # Parse arguments
    args = get_args()
    rosbag_file = os.path.join(args.fDir, args.rosbag_name)

    ##################################
    # Local configuration for debug

    # # General setting
    # args.save_to_file = True
    # args.verbose = True
    args.force_restart = True

    # # For more modules setting
    args.survelliance_system = True
    args.puzzle_solver = True
    args.state_analysis = True
    args.activity_interpretation = True

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
    args.display = '110001' # @< For most debug purposes on puzzle solver
    # args.display = '000001' # @< For most debug purposes on activity analysis


    ###################################

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

        timestamp_beginning= info_dict['start']
        timestamp_ending = info_dict['end']

        print('Playback the rosbag recordings')

        # Need to start later for initialization
        # May need to slow down the publication otherwise the subscriber won't be able to catch it
        # -d:delay; -r:rate; -s:skip
        command = "rosbag play {} -d 2 -r 1 -s 10 --topic {} {} {}".format(
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
