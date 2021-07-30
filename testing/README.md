# Test Data format

- ```[DataName].avi``` or ```[DataName].png```: The RGB video or image
- ```[DataName].npz```: The depth frame(s) or the intrinsic camera matrix, with the keys:
  - "depth_frame(s)": Depth frame(s). If the corresponding ```[DataName]``` is a video, then has the plural 's'. Otherwise does not.
  - "intrinsics": the Intrinsic matrix

