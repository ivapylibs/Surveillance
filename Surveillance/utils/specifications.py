#============================= specifications ============================
"""
  @brief    Specifications loader from a yaml file.
        
  @author   Patricio A. Vela,   pvela@gatech.edu
  @date     2022/07/01

  @note     First time creating.  Maybe there is a cleaner way, but have 
            limited python abilities. Eventually have inheritance 
            structure based on components, if makes sense, or queries
            to known components. For now having stand-alone.

"""
#============================= specifications ============================

import os

class dict2struct:
  def __init__(self, **entries):
    self.__dict__.update(entries)

class specifications:
  def __init__(self, entries):
    self.source  = dict2struct(**entries['source'])
    self.general = dict2struct(**entries['general'])
    self.exec    = dict2struct(**entries['exec'])

    self.source.rosbag = os.path.expanduser(self.source.rosbag)
    self.source.puzzle = os.path.expanduser(self.source.puzzle)

    self.exec.read_activity = False
