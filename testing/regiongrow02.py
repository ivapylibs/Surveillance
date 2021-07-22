"""
=============================== regiongrow01 ============================

    @brief      Test the regiongrow algorithm on a simple synthetic image
                start from a mask
    
    
    @author: Yiye Chen          yychen2019@gatech.edu
    @Date: 07/22/2021 

=============================== regiongrow01 ============================
"""

# ===== environement setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

fPath = os.path.dirname(__file__)
rPath = os.path.dirname(fPath)
uPath = os.path.join(rPath, "utils")
sys.path.append(rPath)

from utils.region_grow import RegionGrower

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
region_grower = RegionGrower()
region_grower.process_mask(fakeI, init_mask)

# ==== show the result
plt.figure()
region_grower.display_results()


plt.show()
