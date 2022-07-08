from yacs.config import CfgNode as _CfgNode

class CfgNode(_CfgNode):
    """ Derived from the yacs CfgNode class to add new features
    """
    def __init__(self, init_dict=None, key_list=None, new_allowed=True):
        super().__init__(init_dict, key_list, new_allowed)

    def merge_from_files(self, cfg_files):
        """Merge from multiple files

        Args:
            cfg_files (list):   A list of cfg file paths
        """
        for file in cfg_files:
            self.merge_from_file(file)
