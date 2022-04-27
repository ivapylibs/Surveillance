"""

@brief          The image/plotting-related utility functions

@author:        Yiye Chen,          yychen2019@gatech.edu
@date:          04/20/2022

"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_contour(bm, img=None, color=(0, 255, 0), thick=3):
    """Draw all the countours of a binary image

    Args:
        bm (array, (H, W)):         The binary map
        img (array, (H, W, 3)):     The RGB image to plot the contour.
                                    If None, then will plot on the binary mask
        color ((3, )):              The RGB color of the contour
        thick (int):                The thickness of the contour in pixel
    Returns:
        im (array, (H, W, 3)):      An RGB image containing the contour
    """

    # get the contour
    bm = np.uint8(bm) * 255
    contours, hierarchy = cv2.findContours(bm,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # plot 
    if img is None:
        rgb_bm = cv2.cvtColor(bm, cv2.COLOR_GRAY2RGB)
        img = rgb_bm
    rgb_contour = cv2.drawContours(
        np.float32(img), contours=contours, contourIdx=-1,
        color=color,
        thickness=thick
    )

    return np.uint8(rgb_contour)

def draw_dots(img, coords, color=[0, 255, 0], radius=20):
    """Draw a set of dots on the image 

    Args:
        img (array): The image to draw the coordinates. Currently only accept rgb input
        coords (N, 2): The coordinates of the dots in the OpenCV style
        color (3,): The rgb color of the dots
        radius (int): The radius of the dots
    Returns:
        img_plot (array): The image with the dots plotted
    """
    if coords is None:
        return img

    coords = np.array(coords)
    N = coords.shape[0]
    img_plot = np.ascontiguousarray(img, dtype=np.uint8)
    for i in range(N):
        center = (int(coords[i, 0]), int(coords[i, 1]))
        img_plot = cv2.circle(img_plot, center, radius, color, -1)

    return img_plot

    
if __name__=="__main__":
    bm = np.zeros((100, 100), dtype=bool)
    bm[20:60, 20:60] = 1
    contour_map = draw_contour(bm, thick=3)
    img = np.round(np.random.rand(bm.shape[0], bm.shape[1], 3) * 255).astype(np.uint8)
    contour_map_2 = draw_contour(bm, img, thick=3)
    # contour_map = bm
    plt.figure()
    plt.imshow(bm)
    plt.title("Input Mask")
    plt.figure()
    plt.imshow(contour_map)
    plt.title("The contour on the binary mask")
    plt.figure()
    plt.imshow(contour_map_2)
    plt.title("The contour on a random rgb image")

    # test the image plot
    dot_coords = np.array(
        [
            [10, 10],
            [10, 20],
            [20, 20]
        ]
    )
    img_dots = draw_dots(img, dot_coords, radius=5)
    plt.figure()
    plt.imshow(img_dots)
    plt.title("THe dot plotted image")

    plt.show()
    