import numpy as np

# The manually measured aruco-to-robot transformation
M_WtoR = np.array(
    [[0, 1, 0, 0.150], 
    [-1, 0, 0, -0.013],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]
)

# @todo     Should probably be in a yaml file. Not hard-coded like this.
