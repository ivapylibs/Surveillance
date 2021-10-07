# ======================================= display =========================================
"""
        @brief: The display-related utility functions
        
        @Author: Yiye Chen                  yychen2019@gatech.edu
        @Date: 10/07/2021

"""
# ======================================= display =========================================

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

if __name__ == "__main__":
    imgSource = lambda : (
        (np.random.rand(100,100,3) * 255).astype(np.uint8), \
        np.random.rand(100,100)
    )
    rgb, dep = imgSource()
    display_rgb_dep(rgb, dep)
    plt.show()
