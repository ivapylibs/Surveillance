import subprocess
import psutil

# to terminate the rosbag record completely. See:https://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
def terminate_process_and_children(p):
    process = psutil.Process(p.pid)
    for sub_process in process.children(recursive=True):
        sub_process.send_signal(subprocess.signal.SIGINT)
    p.wait()  # we wait for children to terminate