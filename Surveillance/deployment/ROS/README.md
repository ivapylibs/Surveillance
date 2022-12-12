# Recorder instructions

This folder contains the scripts for recording the necessary data (calibration + test) into a rosbag file and running the script on the pre-saved data.

## Data

We have captured some testing data in the rosbag format. All the data is stored on the Dropbox. Please download the data from the [Dropbox link](https://www.dropbox.com/sh/6odjfrw522yko09/AAATSuZU7pl4vfzc-lVCro07a?dl=0), and put it in the ```deployment/ROS/data```.

## Usage

Before running any script, navigate to the ```Surveillance/deployment/ROS``` folder first:

```bash
cd path/to/clone/Surveillance
cd Surveillance/deployment/ROS
```

Note if you want to use pycharm with ROS support, you need to start the pycharm program within the terminal already enabling ROS environment, otherwise the ROS python library will not be found.


###  1. Record data

Record both the system calibration data and the test-time rgb and depth data. 

#### 1.1 First-time Calibration

For the first time using the system or want to re-calibrate the system, run:

```bash
python rosbag_data_recorder.py --save_dir ./
```

Where the ```--save_dir``` accepts the destination folder to save the recorded data. If omitted, the default will be ```./```(the same folder as the recorder script.)

The recorder will save a bag file with the date appended: ```data_{RECORDING_DATE_AND_TIME}.bag``` . **Please use ```ls``` to check the data name, and compress it using the following commands (with the ```{RECORDING_DATE_AND_TIME}``` replaced by the actual one**. Without compression the data size will be large:

```bash
rosbag compress data_{RECORDING_DATE_AND_TIME}.bag
rm data_{RECORDING_DATE_AND_TIME}.orig.bag
```
If the purpose is to save the data as calibration data for future usage, it is better to change the prefix to ``calib`` instead of ``data`` to differentiate from recordings used for other purposes. 

Note that we have to be careful when calibrating. The survelliance system is sensitive to the table mat color. So we should keep consistent if there is a user sitting in front of the machine for both calibration and testing.


#### 1.2 Record with the calibrated system

For using the pre-saved calibration data and only record the test-time data, toggle on the option and provide with the calibration data file name:

```bash
python rosbag_data_recorder.py --save_dir ./ --load_exist --rosbag_name path/to/folder/calib_{RECORDING_DATE_AND_TIME}.bag
```

It will fetch the calibration data in the provided rosbag file, record new test-time data, and store them together in a new rosbag file. Then again compress the data following the previous steps.



#### 1.3 Modify the parameters.

A set of parameters are specified in the ```config/NAME.yaml``` file. The rosbag data recorder requires two of them, including the ```config/setup.yaml``` and ```config/ros.yaml```. The previous is for the system setup such as the camera parameters and the Aruco tag size, and the second specify the ros topic names.

To change the parameters (e.g. The camera exposure and gain), the simple way is to modify the parameters in the yaml file. A better way for code development is to copy paste the desired file and pass them to the python script as the argument. Parameter files are seperated using comma (w/o space):

```bash
python rosbag_data_recorder.py --yfiles "config/NAME1.yaml,config/NAME2.yaml,..."
```





### 2. Run the surveillance system 

To build up the surveillance system (with calibration data), please provide the rosbag file name and run the following script:

```bash
python rosbag_runner.py --rosbag_name path/to/folder/data_{RECORDING_DATE_AND_TIME}.bag
```

By default, the test data is assumed from the same rosbag. To work on other sources, please enable the real_time option by ```--real_time```.

Different display options are provided, please input a decimal number or a binary number. Refer to the arguments in ```Surveillance/deployment/ROS/rosbag_runner.py``` for more detail.



### 3. Convert the recorded rosbag to the video file.

The script ```rosbag_to_avi.py``` script is for converting the color frame recorded in a rosbag data to the avi video file. The way to use it (should replace the ```PATH/TO/ROSBAG.bag``` to your actual rosbag path (e.g. ```./data/record.bag```):

```bash
python rosbag_to_avi.py PATH/TO/ROSBAG/NAME.bag
```

It will save the video in the save folder as the rosbag, with the same file name, but with a different extension. For the above example, the saved video will be as ```PATH/TO/ROSBAG/NAME.avi```.

The script will by default "watch" the recording simultaneously when converting to the video. To disable the playback, use the ```--no_vis``` option:

```bash
python rosbag_to_avi.py PATH/TO/ROSBAG/NAME.bag --no_vis
```





