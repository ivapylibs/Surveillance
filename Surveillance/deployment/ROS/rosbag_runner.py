"""

    @brief          The test script that run the system on a rosbag file,
                    include building the system and test the system.

    @author         Yiye Chen,          yychen2019@gatech.edu
                    Yunzhi Lin,         yunzhi.lin@gatech.edu
    @date           02/25/2022

"""

import numpy as np
import os
import subprocess
import yaml
import threading
import time
import cv2
import argparse
import copy

import rospy
import rosgraph
import rosbag

from ROSWrapper.subscribers.Images_sub import Images_sub
from camera.utils.display import display_images_cv, display_rgb_dep_cv

from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams
from Surveillance.deployment.utils import terminate_process_and_children

# puzzle stuff
from puzzle.runner import RealSolver

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
    # data/Testing/data_2022-03-01-18-46-00.bag
    parser.add_argument("--fDir", type=str, default="./", \
                        help="The folder's name")
    parser.add_argument("--rosbag_name", type=str, default="data/Testing/Yunzhi_test/data_2022-03-09-16-16-48.bag", \
                        help="The rosbag file name")
    
    args = parser.parse_args()
    return args

class ImageListener:
    def __init__(self):

        # Data captured
        self.RGB_np = None
        self.D_np = None

        self.rgb_frame_stamp = None
        self.rgb_frame_stamp_prev = None

        # Initialize a node
        rospy.init_node("test_surveillance_on_rosbag")

        # == [0] build from the rosbag
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
            depth_scale_topic="depth_scale"
        )
        self.surv = BaseSurveillanceDeploy.buildFromRosbag(rosbag_file, configs_surv)

        self.puzzleSolver = RealSolver()

        # Initialize a subscriber
        Images_sub([test_rgb_topic, test_dep_topic], callback_np=self.callback_rgbd)

        print("Initialization ready, waiting for the data...")

    def callback_rgbd(self, arg_list):

        print("Get to the callback")

        RGB_np = arg_list[0]
        D_np = arg_list[1]
        rgb_frame_stamp = arg_list[2].to_sec()

        # np.integer includes both signed and unsigned, whereas the np.int only includes signed
        if np.issubdtype(D_np.dtype, np.integer):
            print("converting the depth scale")
            D_np = D_np.astype(np.float32) * self.surv.depth_scale

        with lock:
            self.RGB_np = RGB_np.copy()
            self.D_np = D_np.copy()
            self.rgb_frame_stamp = copy.deepcopy(rgb_frame_stamp)


    def run_system(self):

        with lock:

            if self.RGB_np is None:
                print('No data')
                return

            RGB_np = self.RGB_np.copy()
            D_np = self.D_np.copy()
            rgb_frame_stamp = copy.deepcopy(self.rgb_frame_stamp)

            # Skip images with the same timestamp as the previous one
            if self.rgb_frame_stamp != None and self.rgb_frame_stamp_prev == self.rgb_frame_stamp:
                return
            else:
                self.rgb_frame_stamp_prev = self.rgb_frame_stamp

            print("Running the Surveillance on the test data")
            self.surv.process(RGB_np, D_np)

            humanImg = self.surv.humanImg
            puzzleImg = self.surv.puzzleImg
            postImg = self.surv.meaBoardImg

            # Display
            display_images_cv([self.RGB_np[:, :, ::-1]], ratio=0.5, window_name="Source RGB")
            display_images_cv([humanImg[:, :, ::-1], puzzleImg[:, :, ::-1]], ratio=0.5, window_name="Separate layers")
            display_images_cv([postImg[:, :, ::-1]], ratio=0.5, window_name="meaBoardImg")
            cv2.waitKey(1)

            # Debug only
            global call_back_num

            # Work on the puzzle pieces

            # Currently, initialize the SolBoard with the very first frame.
            # We can hack it with something outside
            if call_back_num==0:
                self.puzzleSolver.setSolBoard(postImg)

            self.puzzleSolver.process(postImg)

            # Display
            display_images_cv([self.puzzleSolver.bMeasImage[:, :, ::-1]], ratio=1, window_name="Measured board")
            display_images_cv([self.puzzleSolver.bTrackImage[:, :, ::-1]], ratio=1, window_name="Tracking board")
            cv2.waitKey(1)

            call_back_num += 1
            print("The processed test frame number: {} \n\n".format(call_back_num))

        global timestamp_ending
        global roscore_proc
        # # Debug only
        print('Current:', rgb_frame_stamp)
        print('Last:', timestamp_ending)

        # We ignore the last 2 seconds
        if timestamp_ending is not None and abs(rgb_frame_stamp - timestamp_ending) < 2:
            rospy.signal_shutdown('Finished')
            # Stop the roscore if started from the script
            if roscore_proc is not None:
                terminate_process_and_children(roscore_proc)

if __name__ == "__main__":

    # Parse arguments
    args = get_args()
    rosbag_file = os.path.join(args.fDir, args.rosbag_name)

    # Start the roscore if not enabled
    if not rosgraph.is_master_online():
        roscore_proc = subprocess.Popen(['roscore'])
        # wait for a second to start completely
        time.sleep(1)

    listener = ImageListener()

    # Get basic info from the rosbag
    info_dict = yaml.safe_load(
        subprocess.Popen(['rosbag', 'info', '--yaml', rosbag_file], stdout=subprocess.PIPE).communicate()[0])
    timestamp_ending = info_dict['end']

    # Need to start later for initialization
    # May need to slow down the publication otherwise the subscriber won't be able to catch it
    command = "rosbag play {} -d 2 -r 1 -s 10 --topic {} {}".format(
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