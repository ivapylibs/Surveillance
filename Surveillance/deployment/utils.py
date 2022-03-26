import subprocess
import psutil
import numpy as np

# To terminate the rosbag record completely or the roscore.
# See:https://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
def terminate_process_and_children(p):
    process = psutil.Process(p.pid)
    for sub_process in process.children(recursive=True):
        sub_process.send_signal(subprocess.signal.SIGINT)
    p.wait()  # we wait for children to terminate

# Convert the depth to before scaling, which is an integer and can thus can save space
def depth_to_before_scale(depth, scale, dtype):
    depth_before_scale = depth / scale
    depth_before_scale = depth_before_scale.astype(dtype)
    return depth_before_scale

# 
def depth_scaled(depth):
    """Return True if the depth is a scaled float data, else False"""
    return False

def isBinary(num):
    """
    @brief  Check if the input is binary or not.

    Args:
        num: The input number

    Returns:
        Whether the input is binary or not.
    """
    for i in str(num):
        if i not in ["0","1"]:
            return False
    return True

def decimalToBinary(n):
    """
    @brief Convert decimal to binary number.

    Args:
        n: input

    Returns:

    """
    return bin(n).replace("0b", "")

def display_option_convert(code):
    """
    @brief Convert the input code into the designed format.

    Args:
        code: The input display option.

    Returns:
        bool_list: a code list of True or False.
    """

    if not isBinary(code):
        # If not Binary
        code = decimalToBinary(int(code))

    # Fill 0
    code = str(code).zfill(6)

    # Currently, we only have 6 options
    if len(code)>6:
        raise RuntimeError('Please give the correct display option following the instruction.')

    # Convert to a bool list
    bool_list = [x=='1' for x in code]

    # Reverse the list to put option 0 first
    bool_list.reverse()

    return bool_list


# rgb preprocessing function
def PREPROCESS_RGB(rgb):
    return rgb


# depth preprocessing function
import matplotlib.pyplot as plt

def PREPROCESS_DEPTH(depth):
    """
    Todo: Not done yet. Still in development.
    """
    return depth

    # plt.ioff()

    # get the zero value map
    depth_missing = (depth == 0)
    depth_missing = depth_missing.astype(np.uint8)
    # plt.figure(1)
    # plt.imshow(depth)
    # plt.figure(2)
    # plt.imshow(depth_missing)

    #### Try the mean filter
    k_size = 5
    kernel = np.ones((k_size, k_size), np.float32) / (k_size ** 2)
    depth = cv2.filter2D(depth, -1, kernel)

    #### Try fill with the left most closest non-zero value
    depth_missing = (depth == 0)
    depth_missing = depth_missing.astype(np.uint8)
    rows, cols = np.nonzero(depth_missing)
    for row, col in zip(rows, cols):
        col_fill = col
        while (True):
            col_fill -= 1
            if col_fill < 0:
                break
            if depth[row, col_fill] != 0:
                depth[row, col] = depth[row, col_fill]
                break

    ##### Try to fill the zero value with the closest non-zero value - TOO slow...
    ## calculate the closest non-zero pixel index
    ## NOTE: said that the setting blow gives "fast, coarse" estimation.
    ## source: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042
    ## The meaning of the labels: non-zero pixels and the closest zero pixels will share the same label
    # dist, labels = cv2.distanceTransformWithLabels(depth_missing, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

    ## get the non-zero pixel(pixel with 0 depth) and fill using the labels
    # rows, cols = np.nonzero(depth_missing)
    # for row, col in zip(rows, cols):
    #    label = labels[row, col]
    #    depth[labels==label] = np.max(depth[labels==label])

    # plt.figure(3)
    # plt.imshow(depth)
    # plt.show()

    return depth

# Non - ROI
def NONROI_FUN(H, W, top_ratio=0.2, down_ratio=0.1):
    nonROI_region = np.zeros((H, W), dtype=bool)
    nonROI_region[:int(H * top_ratio), int(H * down_ratio):] = 1
    return nonROI_region

# Activity codebook
    