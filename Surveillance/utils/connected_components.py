"""

    @brief      The functions related to the conencted components

    @author     Yiye Chen,              yychen2019@gatech.edu
    @date       09/17/2021

"""

# ======== [0] Environment and dependencies
import numpy as np
from skimage.measure import label

# ======== [1] Functions

def getLargestCC(mask):
    """
    Return the largest connected component of a binary mask
    If the mask has no connected components (all zero), will be directly returned

    @param[in]      mask                    The input binary mask
    @param[out]     mask_largestCC          The binary mask of the largest connected component
                                            The shape is the same as the input mask
    """
    labels = label(mask)
    if labels.max() == 0:
        Warning("The input mask has no connected component. \
            Will be directly returned")
        largestCC = mask
    else:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
