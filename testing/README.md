# Test Data Download

All the data is stored on the Dropbox. Please download the data from the [Dropbox link](https://www.dropbox.com/sh/6t9v2vjswof2rk6/AAB-JkFemQqOaeCCFPJi7loda?dl=0), and put it in the ```testing/data```.

Test Data format:
- ```[DataName].avi``` or ```[DataName].png```: The RGB video or image
- ```[DataName].npz```: The depth frame(s) or the intrinsic camera matrix, with the keys:
  - "depth_frame(s)": Depth frame(s). If the corresponding ```[DataName]``` is a video, then has the plural 's'. Otherwise does not.
  - "intrinsics": the Intrinsic matrix

# Unit Test

1. Hand calibration   
(The calibration process has been integrated into the test process.) 
- **humanSG01.py**: Calibration on a colored glove image for training the single Gaussian model.
- **height01.py**: Calibration on a table-top image to build up the height model then the depth range for the arm can be decided.

2. Trackpoint determination  
(Tested in the sub-module)

3. Hand color-based segmentation  
- **humanSG01.py**: A color-based segmentation algorithm for the hand glove via single Gaussian modeling & hand mask center as the tracking point.  
- **humanSG02.py**: Add a region grow postprocessing algorithm on the depth input (The idea is abandoned in the end).  
- **humanSG03.py**: Replace the region grow algorithm with hysteresis thresholding and connected component analysis.  
- **humanSG04.py**: Re-organize code structure.  
- **humanSG05_imgDiff.py**: Add a glove detector based on consecutive image difference   
    
4. Robot detection  
(Left for future)


