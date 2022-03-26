"""

    @brief:     The rosbag data recorder records the calibration data and test data into a single rosbag.
                It won't run the system on the test data

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           02/24/2022

"""

import subprocess
import time
import argparse 
import os
import cv2

import rospy
import rosgraph

from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams
from Surveillance.deployment.utils import terminate_process_and_children

def get_args():
    parser = argparse.ArgumentParser(description="The data recorder that records the Surveillance calibration data and the test data.")
    parser.add_argument("--force_restart", action='store_false', \
                    help="Whether force to restart the roscore.")

    parser.add_argument("--load_exist", action='store_true', \
                        help="""Avoid recalibrating the Surveillance system and load the calibration data from the existing rosbag file. \
                            Need to also provide with the path of the file via the --exist_rosbag_name argument""")
    parser.add_argument("--rosbag_name", type=str, default=None, \
                        help ="The rosbag file name that contains the system calibration data")
    parser.add_argument("--act_collect", action="store_true", \
                        help = "Enable the labelling of the activity using the keyboard. Instruction will be provided in the terminal")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # parse the arguments, and the rosbag name if necessary
    args = get_args()

    if args.force_restart:
        subprocess.call(['killall rosmaster'], shell=True)
        subprocess.call(['killall rosbag'], shell=True)

    if args.load_exist:
        assert args.rosbag_name is not None, "Please provide the rosbag name if with to load from the exist calibration data."
        fDir = "./"
        bag_path = os.path.join(fDir, args.rosbag_name)
    else:
        bag_path = None
    
    if args.force_restart:
        subprocess.call(['killall rosmaster'], shell=True)
        subprocess.call(['killall rosbag'], shell=True)
    
    # start the roscore if necessary
    roscore_proc = None
    if not rosgraph.is_master_online():
        roscore_proc = subprocess.Popen(['roscore'])
        # wait for a second to start completely
        time.sleep(1)
    
    # init core
    rospy.init_node("Surveillance_Data_Recorder")

    # == [0] Start the rosbag recording 
    rosbag_name = "data.bag"
    command = "rosbag record -a -o {}".format(rosbag_name)
    rosbag_proc = subprocess.Popen(command, shell=True)
    time.sleep(1)

    # try:
    # == [1] Build
    configs = bParams(
        markerLength = 0.08,
        W = 1920,               # The width of the frames
        H = 1080,                # The depth of the frames
        reCalibrate = (not args.load_exist),
        ros_pub = True,         # Publish the test data to the ros or not
        test_rgb_topic = "test_rgb",
        test_depth_topic = "test_depth",
        activity_topic= "test_activity",
        visualize = True,
        run_system=False,        # Only save, don't run
        activity_label = args.act_collect
    )
    data_collector = BaseSurveillanceDeploy.buildPub(configs, bag_path=bag_path)
    a = 1

    print("===========Calibration finished===========")
    print("\n")
    print("Press \'c\' to start the recording and Press \'q\' to stop the recording")

    # == [2] Run
    data_collector.run()
    # except:
        # print("Something wrong with the recorder. Terminate all the processes.")

    # == [3] End the recording and compressing
    terminate_process_and_children(rosbag_proc)

    # == [4] Stop the roscore if started from the script
    if roscore_proc is not None:
        terminate_process_and_children(roscore_proc)
