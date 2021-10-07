# ======================================= display =========================================
"""
        @brief: The display-related utility functions
        
        @Author: Yiye Chen                  yychen2019@gatech.edu
        @Date: 10/07/2021

"""
# ======================================= display =========================================

from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

def display_rgb_dep(rgb, depth, suptitle=None, figsize=(10,5), fh=None):
    """Display the rgb and depth image in a same figure at two different axes
    The depth will be displayed with a colorbar beside

    Args:
        rgb (np.ndarray): The rgb image
        depth (np.ndarray): The depth map
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

def wait_for_confirm(color_dep_getter:Callable, color_type="rgb", 
        instruction="Press any key to confirm", fh=None):
    """An interface function for letting the user select the desired frame \
        from the given sensor source. The function will display the color and the depth \
        information received from the source, and then wait for the user to confirm via keyboard. 
    
    NOTE: can't distinguish different keys for now. So only support "press any key to confirm"

    Args:
        color_dep_getter (Callable): The color and depth source. \
            Expect to get the sensor information in the np.ndarray format via: \
                        color, depth = color_dep_getter() \
            When there is no more info, expected to return None
        color_type (str): The color type. RGB or BGR. Will be used for visualization
        instruction ([type], optional): The instruction text printed to the user for selection. Defaults to None.
        fh (plt.Figure, optional): The figure handle. Defaults to None, in which case a new figure will be created

    Returns:
        color [np.ndarray]: The color image confirmed by the user
        depth [np.ndarray]: The depth frame confirmed by the user
    """
    # get the next stream of data
    color, dep = color_dep_getter()

    # prepare the figure
    if fh is None:
        fh = plt.figure()

    # get started
    while((color is not None) and (dep is not None)):

        # visualization 
        display_rgb_dep(color, dep, suptitle=instruction, fh=fh)
        plt.draw()
        plt.show(block=False)

        # wait for confirm
        press_flag = plt.waitforbuttonpress(0.01)
        if press_flag is True:
            plt.close()
            break
        
        # if not confirm, then go to the next stream of data
        color, dep = color_dep_getter()

    return color, dep

if __name__ == "__main__":
    imgSource = lambda : (
        (np.random.rand(100,100,3) * 255).astype(np.uint8), \
        np.random.rand(100,100)
    )
    fh = plt.figure()
    color, dep = wait_for_confirm(imgSource, color_type="rgb", instruction="This is just a dummy example. Press any key for a try", \
        fh=fh)
    # visualize
    display_rgb_dep(color, dep, suptitle="The selected sensor info from the  wait_for_confirm example")
    plt.show()
    