#!/user/bin/python
#============================== regiongrow02 =============================
## @file
# 
# @brief    Test regiongrow algorithm on a simple synthetic image starting
#           from a single pixel.
#
# This code is associated to v1.0 of the Puzzlebot processing.  There have
# been several updates that may render this code out-dated.
#
# @note     Unsure what distinguishes this from `regiongrow01`. Need to review. PAV.
#     
# @ingroup  TestSurveillance_Dev_v1
#
# @version  v1.0 of Puzzlebot
# @author   Yiye Chen          yychen2019@gatech.edu
# @date     2021/07/22
# 
# @quitf
#============================== regiongrow02 =============================

"""
=============================== regiongrow02 ============================

    @brief      Test the regiongrow algorithm on a simple synthetic image
                start from a mask
    
    
    @author: Yiye Chen          yychen2019@gatech.edu
    @Date: 07/22/2021 

=============================== regiongrow02 ============================
"""

# ===== environement setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

fPath = os.path.dirname(os.path.abspath(__file__))

from Surveillance.utils.region_grow import RG_Params
from Surveillance.utils.region_grow import RegionGrower_ValDiff as RegionGrower

# ==== create synthetic data
fakeI = np.zeros((256,256), dtype=np.uint8)
fakeI[100:200, 100:200] = 255
init_mask = np.zeros((256, 256), dtype=bool)
init_mask[130:170, 130:170] = True
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(fakeI, cmap='gray')
ax1.set_title("Original Grayscale image")
ax2.imshow(init_mask, cmap='gray')
ax2.set_title("Initial foreground mask")

# ==== Apply the region grow algorithm
region_grower = RegionGrower(RG_Params)
region_grower.process_mask(fakeI, init_mask)

# ==== show the result
plt.figure()
plt.title("The region grow algorithm result")
region_grower.display_results()

print("\n\n Should see a region grow result that is visually the same as the original synthetic grayscale image")


plt.show()

#
#============================== regiongrow02 =============================
