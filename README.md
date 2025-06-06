# Surveillance

The surveillance system for SuperviseIt project. The objective of
the surveillance system is to keep track of the puzzle player's
hand, the robotic arm, and the puzzle workspace.  Segmentation of
the robotic arm is not for tracking purposes, but for ignoring that
area and knowing that it visually occludes the puzzle workspace.
The robotic arm is under control, thus its state is known. In
principle, the surveillance system collects the data generated by a
perceiver and farms it off for additional processing. It's design
can be monolithic, in which case all of the code is contained
within the surveillance class. This case is most compatible with a
traditionally coded and executed system. It can also be federated,
in which case the manages the flow of information to asynchronously
run processes. Federated implementations are more compatible with
ROS-type implementation that coordinate information flow and
processing through the use of topics.

The modules associated to this surveillance system are:

- Layered segmentation system
  - Extracts puzzle player's hand.
  - Extracts robot arm.
  - Extracts puzzle pieces on the workspace.
  - Everything else that remains is the workspace background layer.
- Hand tracker and action parsing
  - Returns action-oriented track point for the hand.
  - Processing track point signal to assign action labels.
- Robot removal and action state
  - Extraction of robot region from processing/parsing.
  - Take given robot commands and converts to action label as needed.
- Activity analysis
  - Post-process hand, robot, and puzzle action states to infer
    activity.
  - This is activty of the player (hand) and impact on puzzle.
  - Includes a progress monitor for the puzzle (obtained from solver).
- Puzzle board and solver
  - Sub-system that keeps track of the puzzle pieces and their
    locations. Maintains status of which are visible and which are
    not visible (in principle, these would be two distinct layers,
    but might be partitioned using state flags instead of disjoint
    lists.
  - A solution board can exist for the puzzle pieces. In this case,
    there is a data association module connecting puzzle pieces to
    their intended solution state.
  - Can contain a planning module that generates solution actions
    for completing the puzzle from an initial incomplete state.


## Install

To implement the entire system requires several custom packages,
each contained in its own repository.  They are listed below. To
run the surveillance system requires installing the repositories
from source:

- [ivapy](https://github.com/ivapylibs/ivapy)
- [Lie](https://github.com/ivapylibs/Lie)
- [camera](https://github.com/ivapylibs/camera)
- [ROSWrapper](https://github.com/ivaROS/ROSWrapper)
- [improcessor](https://github.com/ivapylibs/improcessor)
- [trackpointer](https://github.com/ivapylibs/trackpointer)
- [detector](https://github.com/ivapylibs/detector)
- [perceiver](https://github.com/ivapylibs/perceiver)
- Surveillance (this repo)
- [puzzle_solver](https://github.com/ivapylibs/puzzle_solver)
- [puzzleSolvers](https://github.com/ADCLab/puzzleSolvers)
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

- [x] Verify that the recording ```ActualTestRecording.bag``` is good.
- [x] Shutdown the visualization of the calibration process when start the recording from the exist calibration file.
- [x] Add another script that convert the rosbag file to the video file for the verification of the recording process.
