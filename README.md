# Surveillance

The surveillance system for SuperviseIt project.

## Install

Install the following repositories from the source:

- [Lie](https://github.com/ivapylibs/Lie)
- [camera](https://github.com/ivapylibs/camera)
- [ROSWrapper](https://github.com/ivaROS/ROSWrapper)
- [improcessor](https://github.com/ivapylibs/improcessor)
- [trackpointer](https://github.com/ivapylibs/trackpointer)
- [detector](https://github.com/ivapylibs/detector)
- [perceiver](https://github.com/ivapylibs/perceiver)
- [puzzleSolvers](https://github.com/ADCLab/puzzleSolvers)
- [puzzle_solver](https://github.com/ivapylibs/puzzle_solver)

```
git clone https://github.com/ivapylibs/Surveillance.git
pip3 install -e Surveillance/
```

## Usage

### Unit test 

Unit test for the individual modules. For more details, refer to [README](testing/README.md).

### System usage

We have integrated the designed system with ROS. For more details, refer to [README](Surveillance/deployment/ROS/README.md).

## TODOs:

04/22/2022 Meeting with the Dr. Adan Vela:

- [ ] Verify that the recording ```ActualTestRecording.bag``` is good.
- [ ] Shutdown the visualization of the calibration process when start the recording from the exist calibration file.
- [ ] Add another script that convert the rosbag file to the video file for the verification of the recording process.
