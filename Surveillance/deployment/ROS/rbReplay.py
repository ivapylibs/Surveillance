#!/usr/bin/python3
"""
    @file           rbReplay.py

    @brief          Replays Surveillance recording rosbag file, video portion only from a
                    yaml file specification. 

    @author         Patricio Vela.      pvela@gatech.edu
    @author         Yiye Chen.          yychen2019@gatech.edu

    @date           2022/07/01

"""

from benedict import benedict
import os

import argparse 
from tqdm import tqdm

import cv2
import rospy
import rosbag
from cv_bridge import CvBridge

from camera.utils.display import display_images_cv 
from Surveillance.utils.specifications import specifications

def get_args():
  parser = argparse.ArgumentParser(description="Playback from Surveillance recording rosbag."
                "The video is displayed to a window.")
  parser.add_argument("yfile", type=str, \
                help="The yaml file containing source data information.")
  args = parser.parse_args()
  return args

def main(rosbag_path):
  
  bridge = CvBridge()                   # the ros-cv converter
  bag = rosbag.Bag(rosbag_path)         # the bag file

  # read the first frame to get the H and W
  for topic, msg, t in bag.read_messages(topics="/test_rgb"):
    rgb = bridge.imgmsg_to_cv2(msg)[:,:,::-1]
    break

  # get frame rate
  num_frames = bag.get_message_count("/test_rgb")
  FPS = 15

  # get started
  print("Replaying rosbag camera video...")

  rgb = None
  bar = tqdm(total=num_frames)
  for topic, msg, t in bag.read_messages(topics="/test_rgb"):
    rgb = bridge.imgmsg_to_cv2(msg)

    display_images_cv([rgb[:,:,::-1]], ratio=0.5, window_name="Raw video from rosbag.")
    opKey = cv2.waitKey(int(1000 * 1./float(FPS)))
    if opKey == ord('q'):
      break
    bar.update(1)

if __name__ == "__main__":
    args = get_args()                           # parse argument. 
    ystr = benedict.from_yaml(args.yfile)       # load yaml file.
    conf = specifications(ystr)

    sfile = os.path.expanduser(conf.source.rosbag)
    print("Loading " + sfile)
    main(sfile)                                 # video replay loop 
