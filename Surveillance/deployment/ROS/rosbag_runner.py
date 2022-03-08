"""

    @brief          The test script that run the system on a rosbag file,
                    include building the model and test the model

    @author         Yiye Chen.          yychen2019@gatech.edu
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

import rospy
import rosgraph
import rosbag

from ROSWrapper.subscribers.Images_sub import Images_sub
from camera.utils.display import display_images_cv, display_rgb_dep_cv

import Surveillance.layers.scene as scene
from Surveillance.deployment.Base import BaseSurveillanceDeploy
from Surveillance.deployment.Base import Params as bParams
from Surveillance.deployment.utils import terminate_process_and_children

# configs - What is the bagfile path, and what is the topic name of the test data and
test_rgb_topic = "/test_rgb"
test_dep_topic = "/test_depth"
fDir = "./"

# prepare
lock = threading.Lock()

timestamp_ending = None
flag_FINISHED = False

# To be built
surv = None
call_back_num = 0


def get_args():
    parser = argparse.ArgumentParser(description="Surveillance runner on the pre-saved rosbag file")
    # data_2022-03-03-16-51-32.bag
    # data_2022-03-02-18-39-29.bag
    parser.add_argument("--rosbag_name", type=str, default="data_2022-03-03-18-18-06.bag", \
                        help ="The rosbag file name that contains the system calibration data")
    
    args = parser.parse_args()
    return args

# TODO: stuck at the visualization for some funny reason. So save the data out for now
def callback(arg_list):

    print("Get to the callback")
    with lock:
        RGB_np = arg_list[0]
        D_np = arg_list[1]
        timestamp = arg_list[2].to_sec()


        # np.integer include both signed and unsigned, whereas the np.int only include signed
        if np.issubdtype(D_np.dtype, np.integer):
            print("converting the depth scale")
            D_np = D_np.astype(np.float32) * surv.depth_scale

        print("Running the Surveillance on the test data")
        surv.process(RGB_np, D_np)

        hImg = surv.humanImg
        pImg = surv.puzzleImg

        # Display Below will also be stucked
        #display_images_cv([hImg[:,:,::-1], pImg[:,:,::-1]], ratio=0.4, window_name="Surv Results")
        #cv2.imshow("Surv Results", hImg)
        #cv2.waitKey(1)

        # Temporary. Save then out
        print("Saving out the data")
        ratio = 0.4
        # images = [hImg[:,:,::-1], pImg[:,:,::-1]]
        images = [RGB_np[:, :, ::-1]]
        H, W = images[0].shape[:2]
        if ratio is not None:
           H_vis = int(ratio * H)
           W_vis = int(ratio * W)
        else:
           H_vis = H
           W_vis = W
        images_vis = [cv2.resize(img, (W_vis, H_vis)) for img in images]
        image_display = np.hstack(tuple(images_vis))

        global call_back_num
        cv2.imwrite("test_frame_{}.png".format(call_back_num), image_display)

        call_back_num += 1
        print("The processed test frame number: {} \n\n".format(call_back_num))

        global timestamp_ending
        global flag_FINISHED

        # # Debug only
        # print('Current:', timestamp)
        # print('Last:', timestamp_ending)

        if timestamp_ending is not None and abs(timestamp - timestamp_ending)<0.1:
            flag_FINISHED = True
            print(flag_FINISHED)

        if flag_FINISHED is True:
            rospy.signal_shutdown('Finished')
            # sys.exit()

if __name__ == "__main__":

    # parse arguments
    args = get_args()
    rosbag_file = os.path.join(fDir, args.rosbag_name)

    # start the roscore if necessary
    roscore_proc = None
    if not rosgraph.is_master_online():
        roscore_proc = subprocess.Popen(['roscore'])
        # wait for a second to start completely
        time.sleep(1)
    rospy.init_node("test_surveillance_on_rosbag")

    # == [0] build from the rosbag
    configs = bParams(
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
    surv = BaseSurveillanceDeploy.buildFromRosbag(rosbag_file, configs)
    
    # Debug only
    # == [1] Prepare the subscribers, and release the test data from the bag
    # check the data - The data has no problem
    # bag = rosbag.Bag(rosbag_file)
    # rgb = None
    # depth = None
    # rgb_num = 0
    # from cv_bridge import CvBridge
    # bridge = CvBridge()
    # plt.ion()
    # plt.show()
    # for topic, msg, t in bag.read_messages([test_rgb_topic, test_dep_topic]):
    #     print(t)
    #     if topic == test_rgb_topic:
    #         rgb = bridge.imgmsg_to_cv2(msg)
    #         rgb_num += 1
    #         print("rgb number: "+str(rgb_num))
    #     elif topic == test_dep_topic:
    #         depth = bridge.imgmsg_to_cv2(msg) * surv.depth_scale
    #     if (rgb is not None) and (depth is not None):
    #         #display_rgb_dep_cv(rgb, depth)
    #         surv.process(rgb, depth)
    #         cv2.waitKey(1)
    #         rgb, depth = (None, None)

    # exit()



    RGB_sub = Images_sub([test_rgb_topic, test_dep_topic],
                         callback_np=callback)
    print("Waiting for the data...")

    # Get basic info
    # Note that https://stackoverflow.com/questions/69564817/typeerror-load-missing-1-required-positional-argument-loader-in-google-col
    info_dict = yaml.safe_load(
        subprocess.Popen(['rosbag', 'info', '--yaml', rosbag_file], stdout=subprocess.PIPE).communicate()[0])

    timestamp_ending = info_dict['end']

    # # Debug only
    # bag = rosbag.Bag(rosbag_file)
    #
    # num = 0
    # for topic, msg, t in bag.read_messages(topics=['/test_rgb']):
    #     print(num)
    #     print('rosbag:',t)
    #     print('header:',msg.header.stamp)
    #     num = num + 1
    # bag.close()
    # # exit()

    # need to slow down the publication or the subscriber won't be able to catch it
    command = "rosbag play {} -d 2 -r 0.5 --topic {} {}".format(
       rosbag_file, test_rgb_topic, test_dep_topic)

    try:
       # Be careful with subprocess, pycharm needs to start from the right terminal environment
       # https://stackoverflow.com/a/3478415
       subprocess.call(command, shell=True)
    except:
       print("Cannot execute the bash command: \n {}".format(command))
       exit()

    rospy.spin()

    # == [4] Stop the roscore if started from the script
    if roscore_proc is not None:
        terminate_process_and_children(roscore_proc)
