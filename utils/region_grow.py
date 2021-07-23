import numpy as np
import matplotlib.pyplot as plt

class RegionGrower():
    def __init__(self, *args, **kwargs):
        self.seed_list = None 
        self.init_mask = None
        self.final_mask = None
        self.reg_avg = 0
        self.reg_var = 0
        # 0: not examined; 1: accepted; -1: discarded
        self.cache_map = None
        self.cache_img = None
        # 8-neighbors
        self.neigb = np.array([[-1, -1], [0, -1], [1, -1], \
                                [-1, 0], [1, 0], \
                               [1, -1], [1, 0], [1, 1]])
        
        # determine the threshold for the the criteria
        if "th_var" in kwargs.keys():
            # the threshold computed from the region variance. th_var * reg_var
            self.th_var = kwargs["th_var"]
        else:
            self.th_var = 5
        if "th_val" in kwargs.keys():
            # direct threshold for the value 
            self.th_val = kwargs["th_val"]
        else:
            self.th_val = 0

    
    def process_seeds(self, img, seeds):
        """
        Region grow from a list of initial seeds

        @param[in] img         The image to apply the algorithm
        @param[in] seeds       (N, 2). A list of the initial seed coordinates (row, col)
        """
        assert len(img.shape) == 2, "Only the 2D img is supported for the current version."
        if isinstance(seeds, list):
            seeds = np.array(seeds)

        # init the seed list,final mask, and the cache map
        self.seed_list = seeds
        self.cache_map = np.zeros_like(img, dtype=int)
        self.cache_map[tuple(seeds.T)] = 1
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
    
    def process_mask(self, img, mask):
        """
        Region grow from an initial mask

        @param[in] img         The image to apply the algorithm
        @param[in] mask        A initial binary mask in numpy. The same shape as the image
        """
        self.init_mask = mask
        seeds_init = np.array([])
        self.process_seeds(img, seeds_init)

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
        """
        check whether a pixel meet the criteria 
        """
        seed_val = self.cache_img[seed[0], seed[1]]
        avg_diff = abs(seed_val - self.reg_avg)
        return avg_diff <= max(
            self.th_var * self.reg_var,
            self.th_val
        )

    