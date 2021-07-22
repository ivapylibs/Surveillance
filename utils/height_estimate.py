# ======================================= z_estimate =========================================
"""
        @brief: Assuming the visual field contains only a surface,
                Given the depth map and the camera intrinsic matrix,
                Derive the transformation to get the distance map of each pixel to the surface
        
        @Author: Yiye Chen                  yychen2019@gatech.edu
        @Date: 07/13/2021

        TODO: move that to elsewhere (improcessor repo?) when done
"""
# ======================================= z_estimate =========================================

import numpy as np

class ZEstimator():
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic
        self.R = None
        self.T = None

    def calibrate(self, depth_map):
        R, T = self._measure(depth_map)
        self._adapt(R, T)
    
    def apply(depth_frame):
        """
        apply the calibrated transformation to a new frame
        """
        return None

    def _measure(self, depth_map):
        """
        measure a new income frame
        """
        R = 0
        T = 0
        return R, T
    def _adapt(self, R, T):
        """
        update the stored transformation
        """
        self.R = None
        self.T = None
