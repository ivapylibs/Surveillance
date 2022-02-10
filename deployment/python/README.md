# System integration test
To test the layered approach, including tabletop/robot/hand/puzzle layers. Different tests focus on different purposes. 

## Test file
- **Puzzle data collector.py**: Add a mask postprocessing part for the puzzle layer.
- **Human puzzle playing.py**: Developed from **Puzzle data collector.py** and add another postprocessing module focusing on the puzzle pieces in the hand-nearby ROI.

## Key parameter
(under if \_name\_=="\_\_main\_\_")
- save_dir: The directory to save the puzzle data.
- save_name:  The image data name pre-fix.
- reCalibrate: Whether recalibrate the system or not. If set as False,  the system will re-use the calibration data saved last time.
- board_type : "test" or "solution", to save the test board or solution board. "Test" expects unassembled data, and "Solution expects a set of assembled pieces. They correspond to different layer post-processing strategies.
- mea_test_r: The radius of circle carved around each piece center for the test-type
- mea_sol_r:  The radius of circle carved around the solution board center for the solution-type
