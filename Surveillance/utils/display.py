# ======================================= display =========================================
"""
        @brief: The display-related utility functions
        
        @Author: Yiye Chen                  yychen2019@gatech.edu
        @Date: 10/07/2021

"""
# ======================================= display =========================================

import matplotlib.pyplot as plt

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
        fh = plt.figure()
    raise NotImplementedError