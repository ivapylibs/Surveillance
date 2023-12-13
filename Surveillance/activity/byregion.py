#=================================== byregion ==================================
"""!
@brief  Activities defined by regions.  Signal must lie in a region to trigger
        associated state.


Should be a general purpose module that triggers activity state when signal enters
known region/zone with a given label.  Whether these zones are mutually exclusive
(disjoint regions) or not is up to the designer.

For now just coding up bare minimum needed.
"""
#=================================== byregion ==================================
"""
@file       byregion.py

@author                Yiye Chen.          yychen2019@gatech.edu
@date                  08/19/2021
"""
#=================================== byregion ==================================

from Surveillance.activity import base as Base


class Planar(activity.base):
    def __init__(self):
        pass

    def process(self, signal):
        """
        Process the new income signal
        """
        self.predict()
        self.measure(signal)
        self.correct()
        self.adapt()

    def predict(self):
        return None

    def measure(self, signal):
        return None

    def correct(self):
        return None
    
    def adapt(self):
        return None


class inImage(activity.base):
    """!
    @brief  Activity states depend on having signal lying in specific regions of an
            image. Signal is presumably in image pixels.

    The presumption is that the regions are disjoint so that each pixel maps to either
    1 or 0 activity/event states.  

    @note Can code both disjoint and overlap.  Disjoint = single image region label.
          Overlapping requires list/array of region masks.  Multiple states can
          be triggered at once. State is a set/list then, not a scalar.
    """

    #========================= inImage / __init__ ========================
    #
    def __init__(self, imRegions = None):
      super(inImage,self).__init__()

      self.imRegions = imRegions

    #============================= setRegions ============================
    #
    def setRegions(self, imRegions):

      # DO YOUR DO.

    #============================== measure ==============================
    #
    def measure(self, signal):
        """
        @brief  Compare signal to expected image region states. 

        @param[in]  signal  The 2D pixel signal value.
        """
        pass


## ADD HDF5 LOAD SUPPORT.
## ADD CALIBRATION SUPPORT.


#
#=================================== byregion ==================================
