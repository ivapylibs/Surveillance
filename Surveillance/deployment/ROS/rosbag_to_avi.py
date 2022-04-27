"""

    @brief:     A script that convert the Surveillance recording rosbag file to the avi video file.

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           04/27/2022

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
from camera.utils.writer import vidWriter

from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams
from Surveillance.deployment.utils import terminate_process_and_children


def get_args():
    parser = argparse.ArgumentParser(description="The converter from the Surveillance recording rosbag file to the avi video file."
                    "The video will be saved to the same directory, with the same name, but with the .avi extention except for the .bag")
    parser.add_argument("rosbag_name", type=str, \
                    help="The path of the rosbag file to be converted")
    parser.add_argument("--no_vis", action='store_true', \
                    help="Watch the frames while converting to the video file.")


    args = parser.parse_args()


    # parse the arguments
    args.visualize = (not args.no_vis)
    return args

def parse_vid_path(rosbag_path, args):
    rosbag_dir = os.path.dirname(rosbag_path)
    rosbag_name = os.path.basename(rosbag_path)
    vid_name = rosbag_name.split('.')[0] + ".avi"
    vid_path = os.path.join(rosbag_dir, vid_name)
    return vid_path

def main(rosbag_path, vid_path, args):

 

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

    # The video write - NOTE: the vidname should not contain extension
    vid_writer = vidWriter(
        dirname=os.path.dirname(vid_path),
        vidname=os.path.basename(vid_path).split('.')[0],
        W = rgb.shape[1],
        H = rgb.shape[0],
        activate=True,
        save_depth=False,
        FPS=FPS
    )

    # get started
    print("The rosbag path: {}".format(rosbag_path))
    print("The save-out video path: {}".format(vid_path))
    print("Converting the rosbag to the video...")

    rgb = None
    bar = tqdm(total=num_frames)
    for topic, msg, t in bag.read_messages(topics="/test_rgb"):
        rgb = bridge.imgmsg_to_cv2(msg)

        # write out
        vid_writer.save_frame(rgb, None)

        # display if required
        if args.visualize:
            display_images_cv([rgb[:,:,::-1]], ratio=0.5, window_name="The recording data from the rosbag.")
            opKey = cv2.waitKey(int(1000 * 1./float(FPS)))
            if opKey == ord('q'):
                break
        bar.update(1)

if __name__ == "__main__":
    # parse the arguments, and the rosbag name if necessary
    args = get_args()

    # parse the video save name
    vid_path = parse_vid_path(args.rosbag_name, args)

    # convert to the video
    main(args.rosbag_name, vid_path, args)

    