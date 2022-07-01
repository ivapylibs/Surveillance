#!/usr/bin/python3
"""

    @brief:         Replays Surveillance recording rosbag file, video portion only. 

    @author         Yiye Chen.          yychen2019@gatech.edu
    @author         Patricio Vela.      pvela@gatech.edu
    @date           07/01/2022

"""

import subprocess
import time
import argparse 
import os
import cv2
from tqdm import tqdm

import rospy
import rosbag
import rosgraph
from cv_bridge import CvBridge

from camera.utils.display import display_images_cv 

def get_args():
    parser = argparse.ArgumentParser(description="Playback from Surveillance recording rosbag."
                "The video is displayed to a window.")
    parser.add_argument("rosbag_name", type=str, \
                help="The path of the rosbag file to be load and replay video from.")
    parser.add_argument("--no_vis", action='store_true', \
                help="Abnormal flag to not visualize but just load bag. For testing purposes.")

    args = parser.parse_args()

    # parse the arguments
    args.visualize = (not args.no_vis)
    return args

def main(rosbag_path, args):

    # the ros-cv converter
    bridge = CvBridge()

    # the bag file
    bag = rosbag.Bag(rosbag_path)

    # read the first frame to get the H and W
    for topic, msg, t in bag.read_messages(topics="/test_rgb"):
        rgb = bridge.imgmsg_to_cv2(msg)[:,:,::-1]
        break

    # get frame rate
    num_frames = bag.get_message_count("/test_rgb")
    FPS = 15

    # get started
    print("The rosbag path: {}".format(rosbag_path))
    print("Replay of rosbag camera video...")

    rgb = None
    bar = tqdm(total=num_frames)
    for topic, msg, t in bag.read_messages(topics="/test_rgb"):
        rgb = bridge.imgmsg_to_cv2(msg)

        # display if required
        if args.visualize:
            display_images_cv([rgb[:,:,::-1]], ratio=0.5, window_name="Raw video from rosbag.")
            opKey = cv2.waitKey(int(1000 * 1./float(FPS)))
            if opKey == ord('q'):
                break
        bar.update(1)

if __name__ == "__main__":
    # parse the arguments, and the rosbag name if necessary
    args = get_args()

    # replay the video
    main(args.rosbag_name, args)

    
