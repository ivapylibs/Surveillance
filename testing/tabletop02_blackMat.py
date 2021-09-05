#!/usr/bin/python3

# ============================ tabletop02_blackMat ==============================
"""
    @brief:         Test whether a black blackground will be constrast the foreground
                    better, thereby reduce the influence of the backgrouned during 
                    the foreground substraction

                    The test will be regarding to two parts:
                    1. Difference between the foreground and the black background. Expected to be bigger
                    2. Difference between the shadow and the black background, expected to be smaller

    @author:    Yiye        yychen2019@gatech.edu
    @date:      09/05/2021
"""
# ============================ tabletop02_blackMat ==============================

# ====== [0] setup the environment. Read the data
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import detector.bgmodel.bgmodelGMM as BG
from Surveillance.layers.human_seg import Human_ColorSG_HeightInRange
from Surveillance.layers.human_seg import Params as hParams

# ======= [0]  setup
fPath = os.path.dirname(os.path.abspath(__file__))
dPath = os.path.join(fPath, 'data/black_mat')


# ======= [1] foreground vs black/white background

# == [1.0] prepare data

# pure background data for frame difference background substraction
black_pure = cv2.imread(
    os.path.join(dPath, "BG_black_0.png")
)[:,:,::-1]
white_pure = cv2.imread(
    os.path.join(dPath, "BG_white_0.png")
)[:,:,::-1]

# foreground(pieces) + background files
piece_blackBG_files = []
piece_whiteBG_files = []
for i in range(5):
    temp_black = os.path.join(dPath, "BG_black_piece_{}.png".format(i))
    temp_white = os.path.join(dPath, "BG_white_piece_{}.png".format(i))
    piece_blackBG_files.append(temp_black)
    piece_whiteBG_files.append(temp_white)

# == [1.1] Define the evaluator

class fgEva():
    """
    @brief      Use the image difference to extract the FG colors and BG colors 
                The image difference is used for fg extraction
                Then the area around the foreground pixels will be assumed to be the background
                Then the color difference will be used to evaluate the FG/BG contrast

    @param[in]  bg_pure         The pure background image for image difference calculation
    @param[in]  th_diff         The threshold for the image difference
    """
    def __init__(self, bg_pure, th_diff=20.):

        # GT puzzle
        self.bg_pure = bg_pure

        # thresholds
        self.th_diff = th_diff  #<the threshold for the image difference

        # store the statistics
        self.fg_colors = np.empty((0, 3))   # (N1, 3)
        self.bg_colors = np.empty((0, 3))   # (N2, 3)
    
    def clear(self):
        self.fg_colors = np.empty((0, 3))   # (N1, 3)
        self.bg_colors = np.empty((0, 3))   # (N2, 3)
    
    def process(self, fg_img, vis=False, fh=None):
        """
        Process a new foregournd image to obtain the new fg_colors and bg_colros

        @param[in]  fg_img          The image containing the foregound on the bg_pure stored
        @param[in]  vis             Visualize the extracted foreground area and the bg area?
        @param[in]  fh              Figure handle
        """
        # get the foreground area
        fgMask = self.get_fg_imgDiff(fg_img)
        bgMask = self.get_bg_around(fgMask)

        # store the statistics
        fg_rows, fg_cols = np.where(fgMask)
        bg_rows, bg_cols = np.where(bgMask)
        fg_colors = fg_img[fg_rows, fg_cols, :] # (N, 3)
        bg_colors = fg_img[bg_rows, bg_cols, :]

        if self.fg_colors.size == 0:
            self.fg_colors = fg_colors
        else:
            self.fg_colors = np.concatenate((self.fg_colors, fg_colors), axis=0)
        if self.bg_colors.size == 0:
            self.bg_colors = bg_colors
        else:
            self.bg_colors = np.concatenate((self.bg_colors, bg_colors), axis=0)

        # visualization
        if vis:
            if fh is None:
                fh = plt.figure()
            axes = fh.subplots(1,3)

            axes[0].imshow(fg_img)
            axes[0].set_title("The original image")
            axes[1].imshow(fg_img * fgMask[:,:,np.newaxis])
            axes[1].set_title("The foreground regions")
            axes[2].imshow(fg_img * bgMask[:,:,np.newaxis])
            axes[1].set_title("The background regions")

    def evaluate(self):
        """
        Evaluate the difference between the stored foreground color and background color 
        """
        pass
    
    def get_fg_imgDiff(self, img):
        """
        Get the foreground mask based on image difference method
        """
        img_diff = np.abs(img.astype(np.float) - self.bg_pure.astype(np.float))
        img_diff = np.mean(img_diff, axis=2) 
        return img_diff > self.th_diff
    
    def get_bg_around(self, fgMask):
        """
        Get the background mask assuming that they are around the fgMask
        """
        pass

    
# == [1.2] Start evaluation

# evaluators
black_evaluator = fgEva(
    black_pure,
    th_diff=20
)

white_evaluator = fgEva(
    white_pure,
    th_diff=20
)

# black
for img_path in piece_blackBG_files:
    img = cv2.imread(img_path)[:,:,::-1]
    black_evaluator.process(img, vis=True)
black_evaluator.evaluate()

# white
for img_path in piece_whiteBG_files:
    img = cv2.imread(img_path)[:,:,::-1]
    white_evaluator.process(img, vis=True)
white_evaluator.evaluate()





# ======= [2] shadow vs black/white background
