#!/usr/bin/python
"""

    @brief:             Test and update the activity analysis runner in a ROS wrapper.
                        Todo: We still have to fix the frame missing issue, in which case, the analysis may be wrong.

    @author:            Yunzhi Lin,          yunzhi.lin@gatech.edu
    @date:              12/4/2022[created]

"""

# ==[0] Prep environment
from dataclasses import dataclass
import copy
import os
from tkinter import Grid
import subprocess
import cv2
import time
import matplotlib.pyplot as plt
import message_filters
import numpy as np
import glob
import argparse
import threading
from std_msgs.msg import String

import rospy
import rosgraph

from puzzle.piece.template import Template, PieceStatus
from puzzle.runner import ParamRunner
from puzzle.runnerROS import RealSolverROS
from puzzle.utils.dataProcessing import convert_ROS2dict

from ROSWrapper.subscribers.Images_sub import Images_sub
from ROSWrapper.subscribers.String_sub import String_sub

from camera.utils.display import display_images_cv

from Surveillance.activity.utils import DynamicDisplay, ParamDynamicDisplay

# configs

postImg_topic = "/postImg"
visibleMask_topic = "/visibleMask"

# Subscribe, the name needs to be consistent with the one in the puzzle solver part
# See https://github.com/ivapylibs/puzzle_solver/tree/yunzhi/puzzle/testing/real_runnerROS.py
puzzle_solver_info_topic = "/puzzle_solver_info"
status_history_topic = "/status_history"
loc_history_topic = "/loc_history"

status_pulse_topic = "/status_pulse"
loc_pulse_topic = "/loc_pulse"

# preparation
lock = threading.Lock()

# To be built
call_back_id = 0

def get_args():
    parser = argparse.ArgumentParser(description="activity analysis runner")
    parser.add_argument("--verbose", action='store_true',
                        help="Whether to debug the system.")
    args = parser.parse_args()

    return args

class ImageListener:
    def __init__(self, opt):

        self.opt = opt

        # Data captured
        self.RGB_np = None
        self.Mask_np = None

        self.rgb_frame_stamp = None
        self.rgb_frame_stamp_prev = None

        # Fig for puzzle piece status display
        self.status_window = None
        self.activity_window = None

        # Data captured
        self.puzzle_solver_info = None
        self.status_history = None  # e.g., {ID: [XX,XX,XX,...], ...}, n_pieces x n_frames
        self.loc_history = None  # e.g., {ID: [array(XX,YY),array(XX,YY),...], ...}, n_pieces x n_frames

        self.status_pulse = None
        self.loc_pulse = None

        # Initialize a subscriber for images
        Images_sub([postImg_topic, visibleMask_topic], callback_np=self.callback_rgbMask)

        String_sub(puzzle_solver_info_topic, String,
                   callback_np=self.callback_puzzle_solver_info)  # Not important for now
        String_sub(status_history_topic, String, callback_np=self.callback_status_history)
        String_sub(loc_history_topic, String, callback_np=self.callback_loc_history)

        String_sub(status_pulse_topic, String, callback_np=self.callback_status_pulse)
        String_sub(loc_pulse_topic, String, callback_np=self.callback_loc_pulse)

        print("Initialization ready, waiting for the data...")

    def callback_rgbMask(self, arg_list):

        if self.opt.verbose:
            print("Get to the callback")

        RGB_np = arg_list[0]
        Mask_np = arg_list[1]
        rgb_frame_stamp = arg_list[2].to_sec()

        with lock:
            self.RGB_np = RGB_np.copy()
            self.Mask_np = Mask_np.copy()
            self.rgb_frame_stamp = copy.deepcopy(rgb_frame_stamp)

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

    def callback_status_pulse(self, msg):

        status_pulse = convert_ROS2dict(msg)

        if len(status_pulse.keys()) > 0:
            status_pulse_processed = {}
            for key in status_pulse.keys():
                if isinstance(key, str):
                    status_pulse_processed[int(key)] = PieceStatus(status_pulse[key])
                else:
                    status_pulse_processed[key] = PieceStatus(status_pulse[key])

            status_pulse = status_pulse_processed

        with lock:
            if self.status_pulse is None:
                self.status_pulse = {}
                for key in status_pulse.keys():
                    self.status_pulse[key] = [status_pulse[key]]
            else:
                for key in status_pulse.keys():
                    self.status_pulse[key].append(status_pulse[key])

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

    def callback_loc_pulse(self, msg):

        loc_pulse = convert_ROS2dict(msg)

        if len(loc_pulse.keys()) > 0:
            loc_pulse_processed = {}
            for key in loc_pulse.keys():
                if isinstance(key, str):
                    loc_pulse_processed[int(key)] = np.array(loc_pulse[key])
                else:
                    loc_pulse_processed[key] = np.array(loc_pulse[key])

            loc_pulse = loc_pulse_processed

        with lock:
            if self.loc_pulse is None:
                self.loc_pulse = {}
                for key in loc_pulse.keys():
                    self.loc_pulse[key] = [loc_pulse[key]] if len(loc_pulse[key])>0 else []
            else:
                for key in loc_pulse.keys():
                    if len(loc_pulse[key])>0:
                        self.loc_pulse[key].append(loc_pulse[key])

    def run_system(self):

        with lock:

            global call_back_id

            if self.RGB_np is None:
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

                # 1) Using the complete status history & loc_history from puzzle solver
                #
                # if self.puzzle_solver_info is not None and self.status_history is not None and self.loc_history is not None:
                #
                #     if self.status_window is None and self.activity_window is None:
                #         self.status_window = DynamicDisplay(
                #             ParamDynamicDisplay(num=self.puzzle_solver_info['solution_board_size'],
                #                                 window_title='Status Change'))
                #         self.activity_window = DynamicDisplay(
                #             ParamDynamicDisplay(num=self.puzzle_solver_info['solution_board_size'],
                #                                 status_label=['NONE', 'MOVE'], ylimit=1,
                #                                 window_title='Activity Change'))
                #
                #     status_data = np.zeros(len(self.status_history))
                #     activity_data = np.zeros(len(self.status_history))
                #
                #     for i in range(len(status_data)):
                #         try:
                #             status_data[i] = self.status_history[i][-1].value
                #             assert status_data[i] == self.status_pulse[i].value
                #         except:
                #             status_data[i] = PieceStatus.UNKNOWN.value
                #
                #         # Debug only
                #         # if len(self.status_history[i])>=2 and \
                #         #         np.linalg.norm(self.loc_history[i][-1] - self.loc_history[i][-2]) > 10:
                #         #     print('!')
                #
                #         if len(self.status_history[i]) >= 2 and \
                #                 self.status_history[i][-1] == PieceStatus.MEASURED and \
                #                 self.status_history[i][-2] != PieceStatus.MEASURED and \
                #                 np.linalg.norm(self.loc_history[i][-1] -
                #                                self.loc_history[i][-2]) > 30:
                #             activity_data[i] = 1
                #             print(f'Move activity detected for piece {i}')
                #
                #         else:
                #             activity_data[i] = 0
                #
                #     self.status_window((call_back_id, status_data))
                #     self.activity_window((call_back_id, activity_data))
                #

                # 2) Only using the status pulse & loc pulse from puzzle solver but save the history on the activity analysis side
                # Yunzhi: the difference is minor
                #
                if self.puzzle_solver_info is not None and self.status_pulse is not None and self.loc_pulse is not None:

                    if self.status_window is None and self.activity_window is None:
                        self.status_window = DynamicDisplay(
                            ParamDynamicDisplay(num=self.puzzle_solver_info['solution_board_size'],
                                                window_title='Status Change'))
                        self.activity_window = DynamicDisplay(
                            ParamDynamicDisplay(num=self.puzzle_solver_info['solution_board_size'],
                                                status_label=['NONE', 'MOVE'], ylimit=1,
                                                window_title='Activity Change'))

                    status_data = np.zeros(len(self.status_pulse))  # n_pieces
                    activity_data = np.zeros(len(self.loc_pulse))  # n_pieces

                    for i in range(len(status_data)):
                        try:
                            status_data[i] = self.status_pulse[i][-1].value
                        except:
                            status_data[i] = PieceStatus.UNKNOWN.value

                        # Debug only
                        # if len(self.status_history[i])>=2 and \
                        #         np.linalg.norm(self.loc_history[i][-1] - self.loc_history[i][-2]) > 10:
                        #     print('!')

                        if len(self.status_pulse[i]) >= 2 and \
                                self.status_pulse[i][-1] == PieceStatus.MEASURED and \
                                self.status_pulse[i][-2] != PieceStatus.MEASURED and \
                                len(self.loc_pulse[i]) >= 2 and \
                                np.linalg.norm(self.loc_pulse[i][-1] -
                                               self.loc_pulse[i][-2]) > 30:
                            activity_data[i] = 1
                            print(f'Move activity detected for piece {i}')

                        else:
                            activity_data[i] = 0

                    self.status_window((call_back_id, status_data))
                    self.activity_window((call_back_id, activity_data))

            if self.opt.verbose:
                print(f"The processed test frame id: {call_back_id} ")

            call_back_id += 1

if __name__ == "__main__":

    args = get_args()

    # Start the roscore if not enabled
    if not rosgraph.is_master_online():
        roscore_proc = subprocess.Popen(['roscore'], shell=True)
        # wait for a second to start completely
        time.sleep(1)

    # Initialize the ROS node
    rospy.init_node('acitivity_analysis')

    plt.ion()
    listener = ImageListener(args)

    while not rospy.is_shutdown():
        listener.run_system()