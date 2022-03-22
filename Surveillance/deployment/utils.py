import subprocess
import psutil

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


    