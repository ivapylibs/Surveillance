"""

    @brief:         The frame transformer handles the coordinate transformations between the image frame (with depth), camera frame, tabletop frame, and the robot frame

    @author:        Yiye Chen
    @date:          11/15/2021

"""

import numpy as np
import cv2
from warnings import warn

class frameTransformer():
    """
    In a world composed of a depth camera, robot, and a defined world frame, 
    the frameTransformer handels the transformation between the image frame, camera frame, tabletop frame, and the robot frame.
    The image and camera frame follow the OpenCV definition

    The transformation matrix required to complish the task:
    1. M_intrinsic: The camera intrinsic matrix, which is the camera-to-image transformation (double direction). P_I = M_int * P_C
    2. M_WtoC: The world-to-camera transformation. One example is using the Aruco tag. P_C = M_WtoC * P_W
    3. M_WtoR: The world-to-robot transformation. P_R = M_WtoR * P_W
    4. M_BEV: The image-to-BEV transformation.  P_BEV = M_BEV * P_I

    Args:
        M_intrinsic ((3,3)): The 3-by-3 intrinsic matrix
        M_WtoC ((4, 4)): The 4-by-4 homogeneous camera-to-world intrinsic matrix
        M_WtoR ((4, 4)): The 4-by-4 homogeneous world-to-robot intrinsic matrix
        BEV_mat ((3, 3)): The Bird-eye-view transformation matrix
    """

    def __init__(self, M_intrinsic=None, M_WtoC=None, M_WtoR=None, M_BEV = None):
        self.M_int = M_intrinsic
        self.M_WtoC = M_WtoC
        self.M_WtoR = M_WtoR
        self.M_BEV = M_BEV
    
    def parsePBEV(self, coords, deps=None):
        """Parse the Bird-eye-view(BEV) coordinates

        Args:
            coords ((N, 2)):        The BEV coordinates 
            deps ((N,), optional):  The deps at the corresponding locations
        Returns
            p_I ((N, 2)):           The image frame coordinates
            p_C [(N, 3)]:           The camera frame coordinates (x, y, z)
            p_W [(N, 3)]:           The world frame coordinates (x, y, z)
            p_R [(N, 3)]:           The robot frame coordinates (x, y, z)
        """
        # The image frame coordinates
        # API Requires the shape (1, N, D). See:https://stackoverflow.com/questions/45817325/opencv-python-cv2-perspectivetransform
        if self.M_BEV is None:
            warn("The BEV matrix is not stored, hence the BEV coordinates can not be parsed. Will return None for all.")
            return None, None, None, None
        p_I = cv2.perspectiveTransform(
            coords[np.newaxis, :, :].astype(np.float32),
            np.linalg.inv(self.M_BEV)
        )[0]

        # the other frame coordinates
        if deps is not None:
            p_C, p_W, p_R = self.parsePImg(p_I, deps=deps)
        else:
            p_C, p_W, p_R = None, None, None

        return p_I, p_C, p_W, p_R


    
    def parsePImg(self, coords, deps):
        """Parse the image frame coordinate and get the coordinate in camera, world, and robot frame if applicable.
        The function requires both the image frame coordinate (u, v) and the depth value w.r.t. the camera.
        Without the depth, one image frame point will corresponds to a line in the 3D space

        Args:
            coords (np.ndarray, (N, 2)): The (u, v) image frame coordinates. N is the query number
            deps (np.ndarray, (N, )): The depth value cooresponds to the coords
        
        Returns:
            p_C [(N, 3)]: The camera frame coordinates (x, y, z)
            p_W [(N, 3)]: The world frame coordinates (x, y, z)
            p_R [(N, 3)]: The robot frame coordinates (x, y, z)
        """
        # (u*z, v*z, z)
        uvz = np.concatenate(
            (coords, np.ones_like(coords[:, :1])),
            axis=1
        )*deps[:, np.newaxis]

        # get the camera frame coordinate
        if self.M_int is None:
            warn("The camera intrinsic matrix is not initiated, so the image frame coordinate can not be transformed to any other frames. Will return None for all")
            return None, None, None
        else:
            p_C = np.linalg.inv(self.M_int) @ (uvz.T)
            p_C = p_C.T
        
        # get the world coordinates
        if self.M_WtoC is None:
            warn("The world-to-camera transformation is not initated. The world and the robot frame can not be get")
            return p_C, None, None
        else:
            p_W = self._transform(np.linalg.inv(self.M_WtoC), p_C, homog=False)

        # get the robot coordinates
        if self.M_WtoR is None:
            warn("The world-to-robot transformation is not initiated. The robot frame coordinates can not be get.")
            return p_C, p_W, None
        else:
            p_R = self._transform(self.M_WtoR, p_W, homog=False)
            return p_C, p_W, p_R

    def parseDepth(self, depth_map):
        """Parse all the pixel coordinates from a depth map

        Args:
            depth_map ((H, W)): The depth map
        Returns:
            pC_map  ((H, W, 3)): The camera frame coordinate
            pW_map  ((H, W, 3)): The world frame coordinate
            pR_map  ((H, W, 3)): The robot frame coordinate
        """
        H, W = depth_map.shape
        coords = self._get_uv(depth_map, vec=True)
        p_C, p_W, p_R = self.parsePImg(coords, depth_map.reshape(H*W,))

        pC_map = p_C.reshape((H, W, 3)) if (p_C is not None) else None
        pW_map = p_W.reshape((H, W, 3)) if (p_W is not None) else None
        pR_map = p_R.reshape((H, W, 3)) if (p_R is not None) else None
        return pC_map, pW_map, pR_map

    def parsePCam(self):
        pass

    def parsePRob(self, robot_coords):
        """Parse the robot frame coordinate and get the coordinate in world, image, and camera frame if applicable.
        Args:
            robot_coords (np.ndarray, (N, 3)): The robot frame coordinates. N is the query number
        
        Returns:
            p_W [(N, 3)]: The world frame coordinates (x, y, z)
            p_C [(N, 3)]: The camera frame coordinates (x, y, z)
            p_Img [(N, 2)]: The image frame coordinates (u, v)
        """
        # get the world frame coordinate
        if self.M_WtoR is None:
            warn("The world-to-robot transformation is not initiated, so the robot frame coordinate can not be transformed to any other frames. Will return None for all")
            return None, None, None
        else:
            p_W = self._transform(np.linalg.inv(self.M_WtoR), robot_coords, homog=False)
        
        # get the camera coordinates
        if self.M_WtoC is None:
            warn("The world-to-camera transformation is not initated. The camera and the image frame coordinates can not be get")
            return p_W, None, None
        else:
            p_C = self._transform(self.M_WtoC, p_W, homog=False)

        # get the image frame coordinates
        if self.M_int is None:
            warn("The camera intrinsic transformation is not initiated. The image frame coordinates can not be get.")
            return p_W, p_C, None
        else:
            p_img = self._transform(self.M_int, p_C, homog=False)
            return p_W, p_C, p_img

    def _transform(self, M_homog, p, homog=False):
        """Homogeneous transformation

        Args:
            M_homog ((D+1, D+1)): The homogeneous transformation
            p ((N, D+1) or (N, D)): The coordinates, either in homogeneous form (D+1) or in the original form (D)
            homog: Return homogeneous form or not
        Returns:
            p_trans ((N, D+1) or (N, D)). The transformed coordinates. (N, D+1) if homog, else (N, D)
        """

        if p.shape[1] == M_homog.shape[0]:
            p_trans_homog = M_homog @ (p.T)
            p_trans_homog = p_trans_homog.T
        elif p.shape[1] == M_homog.shape[0] - 1:
            p_trans_homog = M_homog @ \
                np.concatenate(
                    (p, np.ones_like(p[:, :1])),
                    axis=1
                ).T
            p_trans_homog = p_trans_homog.T
        else:
            raise NotImplementedError

        if homog:
            return p_trans_homog
        else:
            return (p_trans_homog[:, :-1])/p_trans_homog[:, -1].reshape((-1, 1))

    def _get_uv(self, img, vec=False):
        """
        Get the pixel coordinates of an input image.
        The origin (0, 0) is the upperleft corner, with right as u and down as vA
       
        @param[in]  img     The input image of the shape (H, W)
        @param[out] vec     Vectorize the outputs? Default is False
        @param[out] uv_map  The (u, v) coordinate of the image pixels. (H, W, 2), where 2 is (u, v)
                            If vec is True, then will be (H*W, 2)
        """
        H, W = img.shape[:2]
        rows, cols = np.indices((H, W))
        U = cols
        V = rows
        uv_map = np.concatenate(
            (U[:, :, None], V[:, :, None]),
            axis=2
        )

        if vec:
            uv_map = uv_map.reshape(-1, 2)

        return uv_map

