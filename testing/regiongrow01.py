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
from Surveillance.utils.region_grow import RegionGrower_ValDiff as RegionGrower

# ==== create synthetic data
fakeI = np.zeros((256,256), dtype=np.uint8)
fakeI[100:200, 100:200] = 255
plt.figure()
plt.imshow(fakeI, cmap='gray')
plt.title("Original Grayscale image")

# ==== Apply the region grow algorithm
region_grower = RegionGrower(RG_Params)
region_grower.process_seeds(
    fakeI,
    np.array([[128, 128]])
)

# ==== show the result
plt.figure()
plt.title("The region grow algorithm result")
region_grower.display_results()

print("\n\nShould see a region grow result that is visually the same as the original synthetic  grayscale image")

plt.show()



