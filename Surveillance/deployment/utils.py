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
    