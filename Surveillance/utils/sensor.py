# ======================================= sensor =========================================
"""
        @brief: The sensor related utility functions
        
        @Author: Yiye Chen                  yychen2019@gatech.edu
        @Date: 10/06/2021

"""
# ======================================= sensor =========================================


from typing import Callable
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    # get started
    while((color is not None) and (dep is not None)):
        # visualization
        if fh is None:
            fh = plt.figure()
        fh.suptitle(instruction)
        ax0 = fh.add_subplot(121)
        ax1 = fh.add_subplot(122)
        ax0.imshow(color)
        ax0.set_title("The color frame")
        ax1.imshow(dep)
        ax1.set_title("The depth frame")
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

if __name__=="__main__":
    imgSource = lambda : (
        (np.random.rand(100,100,3) * 255).astype(np.uint8), \
        np.random.rand(100,100)
    )
    fh = plt.figure()
    color, dep = wait_for_confirm(imgSource, color_type="rgb", instruction="This is just a dummy example. Press any key for a try", \
        key='s', fh=fh)
    # visualize
    fh, axes = plt.subplots(1,2)
    fh.suptitle("The selected sensor info from the  wait_for_confirm example")
    axes[0].imshow(color)
    axes[0].set_title("The color frame")
    axes[1].imshow(dep)
    axes[1].set_title("The depth frame")
    plt.show()