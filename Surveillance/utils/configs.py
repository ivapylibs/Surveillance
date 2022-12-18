from sre_compile import isstring
from tabnanny import verbose
from xml.etree.ElementTree import QName
from yacs.config import CfgNode as _CfgNode
import sys, os

class CfgNode(_CfgNode):
    """ Derived from the yacs CfgNode class to add new features
    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=True):
        super().__init__(init_dict, key_list, new_allowed)

    def merge_from_files(self, config_files):
        """Merge from multiple files

        Args:
            cfg_files (str or list):    One configuration file, or a list of cfg file paths
        """
        if isinstance(config_files, str):
            config_files = [config_files]

        for file in config_files:
            self.merge_from_file(file)
    
    def load_defaults(self):
        """Load default parameters from the configuration file. To be overwritten"""
        pass

class CfgNode_Surv(CfgNode):

    def load_defaults(self, verbose=False):

        # ====================== Load defaults
        default_file_Surv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deployment/ROS/config/SystemDefaults.yaml")
        if verbose:
            print(f"Loading default parameters from: {default_file_Surv}")
        self.merge_from_file(default_file_Surv)

        # ====================== Preprocess


class CfgNode_SurvRunner(CfgNode_Surv):

    def load_defaults(self, verbose=False):

        # ====================== Load defaults
        super().load_defaults(verbose=verbose)
        default_file_runner = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deployment/ROS/config/testPuzzleSolver01.yml")
        if verbose:
            print(f"Loading default parameters from: {default_file_runner}")
        self.merge_from_file(default_file_runner)

        # ====================== Preprocess
        # Expand "~" if found in the source paths.
        #
        if 'rosbag' in self.source:
          self.source.rosbag = os.path.expanduser(self.source.rosbag)
        if 'puzzle' in self.source:
          self.source.puzzle = os.path.expanduser(self.source.puzzle)
    

if __name__ == "__main__":

    cfg = CfgNode_Surv()
    cfg.load_defaults(verbose=True)
    print(f"Loaded default camera exposure: {cfg.Camera.exposure}")


    # test overwriting load default
    print("\n\n")
    cfg3 = CfgNode_SurvRunner()
    cfg3.load_defaults(verbose=True)
    print(f"Loaded parent camera exposure config: {cfg3.Camera.exposure}")
    print(f"Loaded and preprocessed child config: {cfg3.source.rosbag}")


        
