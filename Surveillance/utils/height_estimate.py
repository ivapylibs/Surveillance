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
from numpy.core.fromnumeric import sort
from sklearn.decomposition import PCA

class HeightEstimator():
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic

        # R: (1, 3). T:(1,). the naming is a little misleading. Consider changing them
        self.R = None
        self.T = None

        # use PCA to fit the 3d plane
        self.pca = PCA(n_components=3)

    def calibrate(self, depth_map):
        R, T = self._measure(depth_map)
        self._update(R, T)
    
    def apply(self, depth_frame):
        """
        apply the calibrated transformation to a new frame
        """
        H, W = depth_frame.shape[:2]
        uv_map = self._get_uv(depth_frame)

        # (H, W, 3). where 3 is (uz, vz, z). Make use of the numpy multiplication broadcast mechanism
        uvz_map = np.concatenate(
            (uv_map, np.ones_like(uv_map[:, :, :1])),
            axis=2
        ) * depth_frame[:, :, None]

        # get the height
        height = self.R @ (uvz_map.reshape((-1, 3)).T) + self.T
        height = height.T.reshape(H, W)
        
        return height 

    def _measure(self, depth_map):
        """
        Measure the transformation parameters from a new training frame

        The plane parameters: (a, b, c, d)
        The intrinsic matrix: M_{int}

        Then the formula is:
        R = (a, b, c) M_{int}^{-1}
        T = -d
        """
        # (H, W, 3)
        p_cam_map = self._recover_p_Cam(depth_map)
        plane_params = self._get_plane(p_cam_map)
        R, T = self._get_RT(plane_params)
        return R, T

    def _update(self, R, T):
        """
        update the stored transformation

        right now just store the new R and T
        """
        self.R = R 
        self.T = T 

    def _get_uv(self, img, vec=False):
        """
        Get the pixel coordinates of an input image.

        The origin (0, 0) is the upperleft corner, with right as u and down as vA
        
        @param[in]  img     The input image of the shape (H, W)
        @param[out] vec     Vectorize the outputs? Default is False

        @param[out] uv_map  The (u, v) coordinate of the image pixels. (H, W, 2), where 2 is (u, v)
        """
        H, W = img.shape[:2]
        rows, cols = np.indices((H, W))
        U = cols
        V = rows
        uv_map = np.concatenate(
            (U[:, :, None], V[:, :, None]),
            axis=2
        )

        # TODO: Vectorize the output instead as a map?
        if vec:
            pass

        return uv_map

    
    def _recover_p_Cam(self, depth_map):
        """
        Recover the 3d camera coordinate with the input depth map and the stored camera intrinsic matrix

        Assuming the 2d frame coordinate used by the intrinsic matrix set the upper left as (0, 0), right is x, down is y.
        This is true for the realsense cameras according to the github issue below:

        https://github.com/IntelRealSense/librealsense/issues/8221#issuecomment-765335642

        @param[out] pC_map       (H, W, 3), where 3 is (xc, yc, zc)
        """
        H, W = depth_map.shape[:2]
        uv_map = self._get_uv(depth_map)

        # (H, W, 3). where 3 is (uz, vz, z). Make use of the numpy multiplication broadcast mechanism
        uvz_map = np.concatenate(
            (uv_map, np.ones_like(uv_map[:, :, :1])),
            axis=2
        ) * depth_map[:, :, None]

        # recover the camera coordinates
        p_Cam_map = np.linalg.inv(self.intrinsic) @ \
            uvz_map.reshape(-1, 3).T
        p_Cam_map = p_Cam_map.T.reshape(H, W, 3)

        return p_Cam_map

    def _get_plane(self, p_Cam_map):
        """
        Estimate a 3D plane from a map of 3d points
        
        TODO: RANSAC or PCA? PCA first. Remove extrame values

        @param[in]  p_Cam_map       (H, W, 3), where 3 is (x_c, y_c, z_c)

        @param[out] plane_param     (4,). (a, b, c, d) that depicts a 3d plane: ax+by+cz+d = 0
        """

        p_Cam_vec = p_Cam_map.reshape((-1, 3))

        # remove extreme value depth, retain only the middle part. Lets be bold and only take the middle 50%. There are too many pixels anyway
        # NOTE: this is different from ivapylibs.improcessor.clipTails. It replace the extreme values with the upper and lower threshold, 
        # where as this one removes them.
        z_c_vec = p_Cam_vec[:, 2]
        N = z_c_vec.size
        sorted_z_c_vec = np.sort(z_c_vec)
        th_low = sorted_z_c_vec[int(N*0.25)]
        th_high = sorted_z_c_vec[int(N*0.75)]
        mask = (z_c_vec >= th_low) & (z_c_vec <= th_high)
        p_Cam_vec_mid = p_Cam_vec[mask, :]

        # fit the plane
        self.pca.fit(p_Cam_vec_mid)
        print("\n Got the fitted plane from the PCA. The three variance ratios are:{} \n".format(self.pca.explained_variance_ratio_))
        normal = self.pca.components_[-1, :] #(a, b, c)
        # use the mean to calculate d
        mean = np.mean(p_Cam_vec_mid, axis=0) 
        d = -normal.reshape((1, -1)) @ mean.reshape((-1, 1))

        plane_param = np.zeros((4,))
        plane_param[:3] = normal
        plane_param[3] = d
        return plane_param
    
    def _get_RT(self, plane_param):
        """
        Get the transformation parameters that transforms a pixel with 1.pixel coordinate 2. depth information
        to the height w.r.t the plane(i.e. the norm direction of the plane) depicted by the plane_param.

        @param[in] plane_param
        """
        R = plane_param[:3].reshape([1,3]) @ np.linalg.inv(self.intrinsic)
        T = plane_param[-1]
        return R, T

        


