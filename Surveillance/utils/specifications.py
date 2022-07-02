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
from benedict import benedict


class dict2struct:
  def __init__(self, **entries):
    self.__dict__.update(entries)

  #============================ __contains__ ===========================
  #
  # @brief  Overload the "in" operator to request whether the class has
  #         the targeted member variable.
  #
  # @param[in]  att_name    The member variable (attribute name).
  #
  def __contains__(self, att_name):
    return hasattr(self, att_name)


class specifications:
  def __init__(self, entries):
    self.source  = dict2struct(**entries['source'])
    self.general = dict2struct(**entries['general'])
    self.module  = dict2struct(**entries['module'])
    self.output  = dict2struct(**entries['output'])

    # Process surveillance system configuration
    #
    if 'surveillance' in entries:
      print('yup, has surveillance')

    self.surveillance  = dict2struct(**entries['surveillance'])

    # Process activity parsing module configuration.
    #
    if 'activity' in entries:
      print('Parsing activity parsing settings.')
      self.activity = dict2struct(**entries['activity'],
                                  **{'read': False})
    else:
      self.activity = dict2struct(**{'read': False})
    # @todo Figure out what "read" is really trying to do and
    #       whether it belongs in the configuration file.

    # Process activity analysis module configuration.
    #
    if 'analysis' in entries:
      print('Parsing activity analysis settings.')
      self.analysis = dict2struct(**entries['analysis'])

    # Process puzzle solver module configuration.
    #
    if 'puzzle' in entries:
      print('Parsing activity analysis settings.')
      self.puzzle = dict2struct(**entries['puzzle'])

    # Expand "~" if found in the source paths.
    #
    if 'rosbag' in self.source:
      self.source.rosbag = os.path.expanduser(self.source.rosbag)
    if 'puzzle' in self.source:
      self.source.puzzle = os.path.expanduser(self.source.puzzle)




  #============================ __contains__ ===========================
  #
  # @brief  Overload the "in" operator to request whether the class has
  #         the targeted member variable.
  #
  # @param[in]  att_name    The member variable (attribute name).
  #
  def __contains__(self, att_name):
    return hasattr(self, att_name)

  @staticmethod
  def load(yfile):
    ystr = benedict.from_yaml(yfile)    # load yaml file.
    print(ystr)
    conf = specifications(ystr)         # constructor specifications instance.
    return conf

