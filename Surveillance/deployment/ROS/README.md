# Recorder instructions

This folder contains the scripts for recording the necessary data (calibration + test) into a rosbag file and running the script on the presaved data.



## Usage

Before running any script, navigate to the ```Surveillance/deployment/ROS``` folder first:

```bash
cd path/to/clone/Surveillance
cd Surveillance/deployment/ROS
```



###  Record data

To record the rgb and depth data during the deployment and the system calibration data (the data required for building the system) into a single ```.rosbag``` file, run the following script then follow the instructions in the terminal:

```base
python rosbag_data_recorder.py
```



When the recording is finished,  it will save a file with the date appended: ```data_{RECORDING_DATE_AND_TIME}.bag``` . **Please use ```ls``` to check the data name, and compress it using the following commands (with the ```{RECORDING_DATE_AND_TIME}``` replaced by the actual one**. Without compression the data size will be large:

```bash
rosbag compress data_{RECORDING_DATE_AND_TIME}.bag
rm data_{RECORDING_DATE_AND_TIME}.orig.bag
```



### Run the surveillance system on the recorded data

To test the surveillance system on the recorded data, **please change the rosbag file name in the line 35 of the script ```rosbag_runner.py```**, then run it by :

```bash
python rosbag_runner.py
```