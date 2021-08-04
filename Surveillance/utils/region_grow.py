"""
========================================== region_grow =================================

        @brief              Contains the classes for the region grow algorithm

        @author             Yiye Chen.              yychen2019@gatech.edu
        @date               07/22/2021 [created]

========================================== region_grow =================================
"""

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class RG_Params():
    """
    The parameters for the region growers

    @param[in]  k_neigb             How many connected pixels to check for each seed. Default=8
    @param[in]  th_val              For value difference based region grower. The threshold for the absolute value difference
    @param[in]  th_var              For value difference based region grower. The threshold for the absolute value difference is obtained by 
                                    seeds variance times the th_var
    """
    k_neigb: int = 8
    
    th_val: float = 5.
    th_var: float = 0.

class RegionGrower_base():
    """
    @brief      The base class for all the region grower class.

    The Region Grow algorithm assumes that a target area is a connected area. 
    The algorithm starts from some initial target pixels (seeds), 
    and then attempts to grow out the whole target region based on some criteria
    
    This base class codes up the region grow algorithm process for extracting the target region:

    1. Initialize from a single pixel or a binary mask, whose pixels will be marked as target pixels 
        and as "seed" (term for the accepted target pixels whose neighbor pixels have not been examined)
    2. while (seed list is not empty):
        2.1     pop out a seed and check its neighborhood pixels.
        2.2     for each neighbor, if it is out of bounadry, or already been marked as accepted or discarded, then pass
        2.3     for each neighbor, if it can pass some criteria, then mark as accepted and add to the seed list
                                    else mark as discarded and continue
    
    """
    def __init__(self, params: RG_Params):
        self.seed_list = None 
        self.init_mask = None
        self.final_mask = None
        self.reg_avg = 0
        self.reg_var = 0
        # 0: not examined; 1: accepted; -1: discarded
        self.cache_map = None
        self.cache_img = None
        # 8-neighbors
        if params.k_neigb == 8:
            self.neigb = np.array([[-1, -1], [0, -1], [1, -1], \
                                    [-1, 0], [1, 0], \
                                   [1, -1], [1, 0], [1, 1]])
        elif params.k_neigb == 4:
            self.neigb = np.array([[-1, 0], [0, 1],
                                [1, 0], [0, -1]])
        # seeds statistics 
        self.reg_avg = 0
        self.reg_var = 0
    
    def process_seeds(self, img, seeds, rejects=[]):
        """
        Region grow from a list of initial seeds

        @param[in] img          The image to apply the algorithm
        @param[in] seeds        (N, 2). 2: (row, col). A list of the initial seed coordinates
        @param[in] rejects      (N, 2). 2: (row, col). A list of discarded pixels that the algorithm shouldn't grow to
        """
        assert len(img.shape) == 2, "Only the 2D img is supported for the current version."
        if isinstance(seeds, list):
            seeds = np.array(seeds)
        if isinstance(rejects, list):
            rejects = np.array(rejects)

        # init the seed list,final mask, and the cache map
        self.seed_list = seeds
        self.cache_map = np.zeros_like(img, dtype=int)
        self.cache_map[tuple(seeds.T)] = 1
        self.cache_map[tuple(rejects.T)] = -1
        self.cache_img = img
        self.final_mask = np.zeros_like(img, dtype=bool)
        self.final_mask[tuple(seeds.T)] = 1
        self._update_reg_stats()

        # start region grow 
        while self.seed_list.size > 0:
            seed = self.seed_list[0]
            self.seed_list = self.seed_list[1:]
            self._process_seed(seed)

        # reset the seed list and the cache map
        self.seed_list = None
        self.cache_map = None
        self.cache_img = None

        return None
    
    def process_mask(self, img, mask, mask_rej=None):
        """
        Region grow from an initial mask

        @param[in] img         The image to apply the algorithm
        @param[in] mask        A initial binary mask in numpy. The same shape as the image
        """
        assert img.shape == mask.shape, \
            "The initial mask must have the same shape as the image, now they are {} and {} separately".format(mask.shape, img.shape)
        self.init_mask = mask
        row_list, col_list = np.where(self.init_mask == 1)
        seeds_init = np.vstack((row_list[None, :], col_list[None, :])).T

        if mask_rej is not None:
            row_list_rej, col_list_rej = np.where(mask_rej == 1)
            rejs_init = np.vstack((row_list_rej[None, :], col_list_rej[None, :])).T
        else:
            rejs_init = []
        
        self.process_seeds(img, seeds_init, rejects=rejs_init)

        return None

    def get_final_mask(self):
        return self.final_mask
    
    def get_init_mask(self):
        if self.init_mask is not None:
            return self.init_mask
        else:
            raise RuntimeError("The processing is not started from a mask, so no initial mask can be provided")
        
    def display_results(self):
        ax = plt.gca() 
        ax.imshow(self.final_mask, cmap="gray")
        return None

    def _process_seed(self, seed):
        """
        process a single seed
        """
        H, W = self.cache_img.shape[:2]
        for delta in self.neigb:
            row_cur = seed[0] + delta[0]
            col_cur = seed[1] + delta[1]
            # inbound? already detected or discard?
            if row_cur < 0 or row_cur >= H or col_cur < 0 or col_cur >= W \
                or self.cache_map[row_cur, col_cur] != 0:
                continue
            # meet criteria or not?
            if self._meet_criteria([row_cur, col_cur]):
                self.cache_map[row_cur, col_cur] = 1
                self.seed_list = np.vstack((self.seed_list, [[row_cur, col_cur]]))
                self.final_mask[row_cur, col_cur] = 1
                self._update_reg_stats()
            else:
                self.cache_map[row_cur, col_cur] = -1

        return None
    
    def _update_reg_stats(self):
        reg_values = self.cache_img[self.final_mask == 1]
        self.reg_avg = np.mean(reg_values)
        self.reg_var = np.var(reg_values)
    
    def _meet_criteria(self, seed):
        raise NotImplementedError("Need to be overwritten by the child classes")
    
class RegionGrower_ValDiff(RegionGrower_base):
    """
        @brief          The value difference based region grower

        The criteria is thresholding the difference between new candidate seed value and the avarage seed value
    """
    def __init__(self, params: RG_Params):
        super().__init__(params)
        
        # determine the threshold for the the criteria
        self.th_val = params.th_val
        self.th_var = params.th_var
    
    def _meet_criteria(self, seed):
        """
        check whether a pixel meet the criteria 
        """
        seed_val = self.cache_img[seed[0], seed[1]]
        avg_diff = abs(seed_val - self.reg_avg)
        return avg_diff <= max(
            self.th_var * self.reg_var,
            self.th_val
        )

    