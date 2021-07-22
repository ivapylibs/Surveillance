import numpy as np

class RegionGrower():
    def __init__(self, *args, **kwargs):
        self.seed_list = []
        self.init_mask = None
        self.final_mask = None
    
    def process_seeds(self, img, seeds):
        """
        Region grow from a list of initial seeds

        @param[in] img         The image to apply the algorithm
        @param[in] seeds       (N, 2). A list of the initial seed coordinates (x, y) w.r.t the bottom left corner
        """
        if isinstance(seeds, list):
            seeds = np.array(list)
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
        return None