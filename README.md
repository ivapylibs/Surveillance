# Surveillance

The surveillance system for the project SuperviseIt

## Install

### Dependencies

Install the following repositories from the source:

- [Lie](https://github.com/ivapylibs/Lie)
- [camera](https://github.com/ivapylibs/camera)
- [ROSWrapper](https://github.com/ivaROS/ROSWrapper)
- [improcessor](https://github.com/ivapylibs/improcessor)
- [trackpointer](https://github.com/ivapylibs/trackpointer)
- [detector](https://github.com/ivapylibs/detector.git)
- [Perceiver](https://github.com/ivapylibs/perceiver)

### Install

```
git clone https://github.com/ivapylibs/Surveillance.git
pip3 install -e Surveillance/
```



## TODOs

System core:

- [ ] ROS publishers of the transformations between the camera, workspace, and the robot
- [ ] ROS publishers of the layers' information: segmentation results and the tracker states.

Rosbag-based Calibration:

- [x] The system builder from the rosbag

Rosbag recorder/runner scripts:

- [ ] Add the automatic start for roscore to the ros deployment scripts
- [x] (Dependent on the Rosbag-based calibration item above) Build from the calibration data in the pre-saved rosbag file, and run on the test data stored in that same rosbag file. i.e. **The runner**
- [ ] (Dependent on the Rosbag-based calibration item above) Build from the pre-saved calibration data, and record both the calibration information and the test data into a new rosbag. 
- [ ] Instructions for the usage of the rosbag recorder and runner

Others:

- [ ] Incorporate the functions in puzzle_data_collector(may leave for puzzle solver)

  

ROS package (Will be implemented in the [Mary_ROS](https://github.gatech.edu/VisMan/Mary_ROS) package.)- Build a package for the Surveillance. Establish different launch files for different functionalities, including:

- [ ] Deployment:
  - [ ] Build from the presaved rosbag file (similar as the first item in the )
  - [ ] Build from the camera source and record the rosbag. ()



Before deliver the recorder:

- Check the dependencies and whether they can be installed by pip, especially the **roscore** and **rosgraph**.
