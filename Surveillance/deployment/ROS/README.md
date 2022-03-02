# Recorder instructions

This folder contains the scripts for recording the necessary data (calibration + test) into a rosbag file and running the script on the presaved data.



## Usage

Before running any script, navigate to the ```Surveillance/deployment/ROS``` folder first:

```bash
cd path/to/clone/Surveillance
cd Surveillance/deployment/ROS
```



###  1. Record data

Record both the system calibration data and the test-time rgb and depth data.

#### 1.1 First-time Calibration

For the first time using the system or want to re-calibrate the system, run:

```bash
python rosbag_data_recorder.py
```

The recorder will save a bag file with the date appended: ```data_{RECORDING_DATE_AND_TIME}.bag``` . **Please use ```ls``` to check the data name, and compress it using the following commands (with the ```{RECORDING_DATE_AND_TIME}``` replaced by the actual one**. Without compression the data size will be large:

```bash
rosbag compress data_{RECORDING_DATE_AND_TIME}.bag
rm data_{RECORDING_DATE_AND_TIME}.orig.bag
```



#### 1.2 Record with the calibrated system

For using the pre-saved calibration data and only record the test-time data, toggle on the option and provide with the calibration data file name:

```bash
python rosbag_data_recorder.py --load_exist --rosbag_name data_{RECORDING_DATE_AND_TIME}.bag
```

It will fetch the calibration data in the provided rosbag file, record new test-time data, and store them together in a new rosbag file. Then again compress the data following the previous steps.



### 2. Run the surveillance system on the recorded data

To test the surveillance system on the recorded data, provide with the rosbag file name and run the following script:

```bash
python rosbag_runner.py --rosbag_name data_{RECORDING_DATE_AND_TIME}.bag
```