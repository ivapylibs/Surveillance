"""
===================================== puzzle_seg ===================================

    @brief          The puzzle segmenters that aims to extract the puzzle layer

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           09/21/2021

===================================== puzzle_seg ===================================
"""

# dependencies
from dataclasses import dataclass
import detector.bgmodel.bgmodelGMM as bgmodelGMM

import Surveillance.layers.base_fg as base_fg

#params definition
@dataclass
class Params_Residual(base_fg.Params):
    """
    @param preprocess: the preprocess of the input image
    @param postprocess: post process of the detected layer mask
    """
    # Any new parameters?
    #
    def __post_init__(self):
        return super().__post_init__()

# classes definition
class Puzzle_Residual(base_fg.Base_fg):
    """
    This class detect the puzzle layer as the residual of the other masks
    """
    def __init__(self, theDetector:bgmodelGMM.bgmodelGMM_cv, params:Params_Residual) -> None:
        super().__init__(theDetector, None, None, params)
    
    @staticmethod
    def build():
        raise NotImplementedError