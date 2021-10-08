# ======================================= display =========================================
"""
        @brief: The display-related utility functions
        
        @Author: Yiye Chen                  yychen2019@gatech.edu
        @Date: 10/07/2021

"""
# ======================================= display =========================================

from typing import Callable, List
import matplotlib.pyplot as plt
import numpy as np
import cv2

def display_rgb_dep_plt(rgb, depth, suptitle=None, figsize=(10,5), fh=None):
    """Display the rgb and depth image in a same figure at two different axes
    The depth will be displayed with a colorbar beside.

    NOTE: This function use the matplotlib for visualization, which is SLOW.
    SO it is not suitable for real-time visualization (e.g. Visualize the camera feed)

    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        depth (np.ndarray, (H, W)): The depth map
        suptitle (str, optional): The suptitle for display. Defaults to None, which will display no suptitle
        figsize (tuple, optional): The figure size following the matplotlib style. Defaults to (10,5).
        fh (matplotlib.Figure): The figure handle. Defaults to None, which means a new figure handle will be created.
    """

    if fh is None:
        fh = plt.figure(figsize=figsize)

    if suptitle is not None:
        fh.suptitle(suptitle)
    ax0 = fh.add_subplot(121)
    ax1 = fh.add_subplot(122)
    # rgb
    ax0.imshow(rgb)
    ax0.set_title("The color frame")
    # depth
    dep_show = ax1.imshow(depth)
    ax1.set_title("The depth frame")
    # depth colorbar. Give it its own axis. See: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    cax = fh.add_axes([ax1.get_position().x1+0.01,ax1.get_position().y0,0.02,ax1.get_position().height])
    plt.colorbar(dep_show, cax=cax) 

def display_images_cv(images:tuple, window_name="OpenCV Display"):
    """Display a sequence of images

    Args:
        images (tuple): A tuple of images
        window_name (str, Optional): The window name for display. Defaults to \"OpenCV display\"
    """
    #  Stack both images horizontally
    image_display = np.hstack(images)
    #  Show images
    cv2.imshow(window_name, image_display)

def display_rgb_dep_cv(rgb, depth, ratio=None, window_name="OpenCV Display"):

    """Display the rgb and depth image using the OpenCV

    The depth frame will be scaled to have the range 0-255.
    The rgb and the depth frame will be resized to a visualization size and then concatenate together horizontally 
    Then the concatenated image will be displayed in a same window.

    There will be no color bar for the depth

    Args:
        rgb (np.ndarray, (H, W, 3)): The rgb image
        depth (np.ndarray, (H, W)): The depth map
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing
        window_name (sting, Optional): The window name for display. Defaults to \"OpenCV display\"
    """
 

    # compute visualization size. Currently normalize so that the width = 640 
    H, W = rgb.shape[:2]
    if ratio is not None:
        H_vis = int(ratio * H)
        W_vis = int(ratio * W)
    else:
        H_vis = H
        W_vis = W

   # scale to 255, convert to 3-channel
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255./depth.max(), beta=0), cv2.COLORMAP_JET)
    # resize
    color_image_show = cv2.resize(rgb, (W_vis, H_vis))
    depth_colormap = cv2.resize(depth_colormap, (W_vis, H_vis) )
    display_images_cv((color_image_show, depth_colormap), window_name=window_name)
    

def wait_for_confirm(color_dep_getter:Callable, color_type="rgb", 
        instruction="Press \'c\' key to confirm", ratio=None):
    """An interface function for letting the user select the desired frame \
        from the given sensor source. The function will display the color and the depth \
        information received from the source, and then wait for the user to confirm via keyboard. 
    
    NOTE: can't distinguish different keys for now. So only support "press any key to confirm"

    Args:
        color_dep_getter (Callable): The color and depth source. \
            Expect to get the sensor information in the np.ndarray format via: \
                        rgb, depth = color_dep_getter() \
            When there is no more info, expected to return None
        color_type (str): The color type. RGB or BGR. Will be used for visualization
        instruction ([type], optional): The instruction text printed to the user for selection. Defaults to None.
        ratio (float, Optional): Allow resizing the images before display.  Defaults to None, which means will perform no resizing

    Returns:
        rgb [np.ndarray]: The rgb image confirmed by the user
        depth [np.ndarray]: The depth frame confirmed by the user
    """
    # get the next stream of data
    rgb, dep = color_dep_getter()

    # get started
    while((rgb is not None) and (dep is not None)):

        # visualization 
        display_rgb_dep_cv(rgb[:,:,::-1], dep, window_name=instruction, ratio=ratio)

        # wait for confirm
        opKey = cv2.waitKey(1)
        if opKey == ord('c'):
            break
        
        # if not confirm, then go to the next stream of data
        rgb, dep = color_dep_getter()

    return rgb, dep

if __name__ == "__main__":
    imgSource = lambda : (
        (np.random.rand(100,100,3) * 255).astype(np.uint8), \
        np.random.rand(100,100)
    )
    color, dep = wait_for_confirm(imgSource, color_type="rgb", instruction="This is just a dummy example. Press the \'c\' key to confirm", \
        ratio=2)
    # visualize
    display_rgb_dep_plt(color, dep, suptitle="The selected sensor info from the  wait_for_confirm example. Use the Matplotlib for display")
    display_rgb_dep_cv(color, dep, window_name="The selected sensor info from the  wait_for_confirm example. Use the OpenCv for display")
    
    cv2.waitKey(1)
    plt.show()
    