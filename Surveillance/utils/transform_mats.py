import numpy as np

# The manually measured aruco-to-robot transformation
M_WtoR = np.array(
    [[0, 1, 0, 0.146], 
    [-1, 0, 0, -0.01],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]
)

