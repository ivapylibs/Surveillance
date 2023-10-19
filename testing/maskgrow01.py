#!/usr/bin/python
"""
=============================== regiongrow01 ============================

    @brief      Test the regiongrow algorithm on a simple synthetic image
                start from a single pixel
    
    @author: Yiye Chen          yychen2019@gatech.edu
    @Date: 07/22/2021 

=============================== regiongrow01 ============================
"""

# ===== environement setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

fPath = os.path.dirname(os.path.abspath(__file__))

from Surveillance.utils.region_grow import RG_Params
from Surveillance.utils.region_grow import MaskGrower 

# ==== create synthetic data
fakeI = np.zeros((256,256), dtype=np.bool)
#fakeI[100:120, 100:120] = True
plt.figure()
plt.imshow(fakeI, cmap='gray')
plt.title("Original Mask")


bigrI = np.zeros((256,256), dtype=np.bool)
bigrI[70:150, 70:150] = True
bigrI[10:10,10:20] = True

# ==== Apply the region grow algorithm
region_grower = MaskGrower(RG_Params)
region_grower.process_mask(bigrI,fakeI)

# ==== show the result
plt.figure()
plt.title("The region grow algorithm result")
region_grower.display_results()

print("\nShould see a region grow result that is visually the same as the original synthetic  grayscale image\n")

plt.show()



