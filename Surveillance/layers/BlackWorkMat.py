#================================= BlackWorkMat ================================
'''!

@brief  Detector for the black mat (planar) workspace.

Follows the structure of the Puzzle Scene perceiver, which packages everything
into one file since python has individual import facilities, and placing in one
uniform location simplifies things.  However, the black mat layer part is
much simpler as it only performs the color segmentation and does not include
depth information.  The depth information usually gets coupled with the black
mat and the red glove to differentiate the two, and the robot arm.  This 
height based depth check is integrated with the Puzzle Scene perceiver.

Code here is copied from the Puzzle Scene background detection class. The reason 
that they were all mashed together in Puzzle Scene is to take advantage of common
image processing and the depth layer information, and to not separate things such 
that efforts to reduce repeated computation make the data passing too complex.  
Changes should be mirrored across these two files.

What should be contained in this file:
    1. Black mat layer detector from RGBD input.
    2. Multi-centroid track pointer for visualizing objects on mat.
    3. A calibration scheme for the entire process with saving to
        YAML and HDF5 files.

This single file replaces/supercedes the existing base_bg file in this 
directory (and by extension possibly others, like tabletop_seg since that
only appears to use the OpenCV GMM method).  

@note   Would be interesting to compare the SGM and GMM approaches.  Maybe
        try later.
'''
#================================= BlackWorkMat ================================

#
# @file     BlackWorkMat.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/29
#
#
# CODE NOTE:    Using 2 spaces for indent (not following PEP style).
#               90 columns view. 8 space right margin.
#
#================================= BlackWorkMat ================================

#--[0.A] Standard python libraries.
#
import numpy as np
import scipy
import cv2
from dataclasses import dataclass
import warnings

import h5py

#from skimage.segmentation import watershed
#import skimage.morphology as morph

#--[0.B] custom python libraries (ivapylibs)
#
import camera.utils.display as display
from camera.base import ImageRGBD

#from Surveillance.utils.region_grow import RG_Params
#from Surveillance.utils.region_grow import MaskGrower

#--[0.C] PuzzleScene specific python libraries (ivapylibs)
#
#from detector.Configuration import AlgConfig
import detector.inImageRGBD as detBase
import detector.bgmodel.inCorner as detBlack
from detector.inImage import detectorState
import trackpointer.centroidMulti as tracker

#import trackpointer.toplines as tglove
#import trackpointer.centroidMulti as tpieces

#import trackpointer.simple as simple

import perceiver.simple as perBase

import puzzle.defaults as defaults

#
#-------------------------------------------------------------------------------
#============================= Configuration Nodes =============================
#-------------------------------------------------------------------------------
#


#============================== CfgDetector ==============================
#
class CfgDetector(detBlack.CfgInCorner):
  '''!
  @brief    Configuration instance for glove tracking detector.  
  '''
  #------------------------------ __init__ -----------------------------
  #
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):
    '''!
    @brief    Instantiate a puzzle scene (black mat) detector.
    '''
  
    if init_dict is None:
      init_dict = CfgDetector.get_default_settings()

    super(CfgDetector,self).__init__(init_dict, key_list, new_allowed)


  #------------------------ get_default_settings -----------------------
  #
  @staticmethod
  def get_default_settings():

    default_settings = detBlack.CfgInCorner.get_default_settings()
    default_settings.update(dict(mask = None))

    return default_settings



#=========================== CfgPuzzlePerceiver ==========================
#
class CfgPuzzlePerceiver(perBase.CfgPerceiver):
  '''!
  @brief    Configuration instance for glove tracking perceiver.  Designed
            to work for processing subsets (detect, track, etc).
  '''
  #------------------------------ __init__ -----------------------------
  #
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):
    '''!
    @brief    Instantiate a puzzle scene configuration object.
  
    '''
  
    if init_dict is None:
      init_dict = CfgPuzzlePerceiver.get_default_settings()

    super(CfgDetector,self).__init__(init_dict, key_list, new_allowed)


#
#-------------------------------------------------------------------------------
#========================= Builder and Helper Functions ========================
#-------------------------------------------------------------------------------
#

def defaultBuildCfg():
  """!
  @brief    Instantiate a typical builder configuration for black mat puzzle 
            perceiver. Only the detector matters.

  @return   A perceiver build configuration.
  """

  bconf = perBase.BuildCfgPerceiver()
  bconf.detector = CfgDetector()

  # The rest are none so that programmed defaults get used.

  return bconf


def defaultBuildCfg_DetectorLoad(detfile):

  bconf = perBase.BuildCfgPerceiver()
  bconf.detector = detfile

  return bconf

#
#-------------------------------------------------------------------------
#================================ Detector ===============================
#-------------------------------------------------------------------------
#

class Detector(detBlack.inCorner):

  def __init__(self, processors=None, bgMod = None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  processors  Image processors for the different layers.
    '''

    super(Detector,self).__init__(processors, bgMod)

    self.mask = None
    self.imFG = None

  #============================== measure ==============================
  #
  def measure(self, I):

    super(Detector,self).measure(I)

    if (self.mask is not None):
      self.imFG = np.logical_and(np.logical_not(self.Ip), self.mask)
    else:
      self.imFG = np.logical_not(self.Ip)

  #============================== setMask ==============================
  #
  def setMask(self, theMask):

    self.mask = theMask
    # @todo Should check dimensions.  For now ignoring.
    # TODO

  #============================== getState =============================
  #
  def getState(self):
    '''!
    @brief      Get the complete detector state, which involves the 
                states of the individual layer detectors.

    @param[out]  state  The detector state for each layer, by layer.
    '''

    cState   = detectorState()
    cState.x = self.imFG 

    return cState

  #================================ info ===============================
  #
  def info(self):

    tinfo = dict(name = 'filename', version = '0.1',
                 date = 'what', time = 'now',
                 CfgBuilder = None)

    return tinfo

    # @todo     Need to flesh out. Duplicate below.
    #tinfo.name = mfilename;
    #tinfo.version = '0.1;';
    #tinfo.date = datestr(now,'yyyy/mm/dd');
    #tinfo.time = datestr(now,'HH:MM:SS');
    #tinfo.trackparms = bgp;

  #=============================== saveTo ==============================
  #
  #
  def saveTo(self, fPtr):
    '''!
    @brief     Save the instantiated Detector to given HDF5 file.

    The save process saves the necessary information to re-instantiate
    a Detector class object.  Stored in root of fPtr.

    @param[in] fPtr    An HDF5 file point.
    '''

    # Recursive saving to contained elements. They'll make their
    # own groups.
    super(Detector,self).saveTo(fPtr)
    if (self.mask is not None):
      fPtr.create_dataset("theMask", data=self.mask)

  #============================= display_cv ============================
  #
  # @brief  Display any found track points on passed (color) image.
  #
  #
  def display_cv(self, I, ratio = None, window_name="foreground objects"):
    
    display.rgb_binary_cv(I, self.imFG, ratio, window_name)


  #
  #-----------------------------------------------------------------------
  #============================ Static Methods ===========================
  #-----------------------------------------------------------------------
  #

  #============================ buildFromCfg ===========================
  #
  @staticmethod
  def buildFromCfg(theConfig, processor=None):
    '''!
    @brief  Build an inCorner instance from an algorithm configuration instance.

    @param[out] bgDet   Instantiated inCorner background model detector.
    '''

    if (theConfig.cutModel == 'Planar'):
      cutModel = PlanarModel.buildFromCfg(theConfig.cutParmsPlanar, theConfig.tau,
                                                           theConfig.isVectorized)

    elif (theConfig.cutModel == 'Spherical'):
      #DEBUG
      print('Yup it is Spherical. Being tested. Delete this print if it works.')
      cutModel = SphericalModel.buildFromCfg(theConfig.cutParmsSpherical, theConfig.tau,
                                                           theConfig.isVectorized)

    else:
      #DEBUG
      print('Something is off!!')
      print(theConfig)
      return None

    bgDet = Detector(processor, cutModel)
    if (theConfig.mask is not None):
      bgDet.setMask(theConfig.mask)

    return bgDet



  #================================ load ===============================
  #
  def load(inFile):
    fptr = h5py.File(inFile,"r")
    theDet = Detector.loadFrom(fptr)
    fptr.close()
    return theDet

  #============================== loadFrom =============================
  #
  def loadFrom(fPtr, processor = None):
    # Check if there is a mask

    gptr = fPtr.get("bgmodel.inCorner")

    bgModel = None
    for name in gptr:
      if   (name == 'PlanarModel'):
        bgModel = detBlack.PlanarModel.loadFrom(gptr)
      elif (name == 'SphericalModel'):
        bgModel = detBlack.SphericalModel.loadFrom(gptr)

    theDetector = Detector(processor, bgModel)
    if (bgModel is None):
      print("Uh-oh: No background inCorner model found to load.")
    else:
      theDetector.set_model(bgModel)

    keyList = list(fPtr.keys())
    if ("theMask" in keyList):
      maskPtr = fPtr.get("theMask")
      wsMask  = np.array(maskPtr)
      theDetector.setMask(wsMask)

    return theDetector

  #========================== calibrate2config =========================
  #
  # @brief  Canned calibration of detector based on layered components.
  #
  # The approach has been tested out using individual test scripts located
  # in the appropriate ``testing`` folder of the ``detector`` package.
  # The starting assumption is that an RGBD streamer has been created
  # and that it provides aligned RGBD images.
  #
  # Since the detection schemes usually rely on an initial guest at the
  # runtime parameters, the presumption is that an approximate, functional
  # configuration is provided.  It is refined and saved to an HDF5 file.
  #
  # Unlike the earlier save/load approaches, this one does not require
  # going through the class member function for saving as the layered system
  # is not fully instantiated.  Not sure if this is a drawback or not.
  # Will code up both versions, then maybe remove one.  One version goes
  # through the layered detector class, the other involves hard coding those
  # same steps within this static member function and never instantiating a
  # layered detector.
  #
  # @param[in] theStream    Aligned RGBD stream.
  # @param[in] outFile      Full path filename of HDF5 configuration output.
  #
  @staticmethod
  def calibrate2config(theStream, outFile, isRGBD=False):

    #==[1]  Step 1 is to get the background color model.
    #       Hardcoded initial configuration with some refinement.
    #
    # @todo Should trace through code to see if this even does anything.
    #
    bgModel    = detBlack.inCorner.build_model_blackBG(-70, 0)
    bgDetector = Calibrator()

    bgDetector.set_model(bgModel)
    if isRGBD:
      bgDetector.refineFromStreamRGBD(theStream, True)
    else:
      bgDetector.refineFromStreamRGB(theStream, True)

    #==[2]  Step 2 is to get the largest region of interest as a 
    #       workspace mask.  Then apply margins generated from refinement
    #       processing in the earlier step.
    #
    if isRGBD:
      theMask = bgDetector.maskRegionFromStreamRGBD(theStream, True)
    else:
      theMask = bgDetector.maskRegionFromStreamRGB(theStream, True)

    kernel  = np.ones((3,3), np.uint8)
    scipy.ndimage.binary_erosion(theMask, kernel, 2, output=theMask)
    bgDetector.setMask(theMask)

    bgDetector.apply_estimated_margins()
    bgDetector.bgModel.offsetThreshold(35)

    # @todo Definitely can be improved.  Masking step and margin
    #       step can be combined.  Margin can be applied universally
    #       across image after averaging in mask region.  Offset
    #       threshold applied as needed.

    #==[3]  Step 3 is to save as a configuration.
    #
    bgDetector.save(outFile)


#
#-------------------------------------------------------------------------
#============================ PuzzlePerceiver ============================
#-------------------------------------------------------------------------
#

@dataclass
class InstPuzzlePerceiver():
    '''!
    @brief Class for collecting visual processing methods needed by the
           Puzzle pieces perceiver. It only works to capture the puzzle 
           pieces and assumes there are no distractor objects.

    '''
    detector : any
    trackptr : any
    trackfilter : any
    #to_update : any    # What role/purpose??

# @todo Push to perceiver class??  Seems to be generic.
# @todo This part not yet worked out.  NEXT UP WHEN IMPROVING THIS CODE.
# IAMHERE



class PuzzlePerceiver(perBase.simple):

  def __init__(self, perCfg = None, perInst = None):

    if perInst is not None:
      super().__init__(perCfg, perInst.detector, perInst.trackptr, perInst.trackfilter)
    else:
      raise Exception("Sorry, not yet coded up.") 
      # @todo   Presumably contains code to instantiate detector, trackptr, filter, etc.
    

  # @note   MOST OF THE METHODS BELOW SHOULD BE THE SAME AS THE BASE PERCEIVER
  #         IMPLEMENTATION AND ARE NOT NEEDED.  REVIEW AND DELETE THOSE THAT ARE NOT
  #         NEEDED.
  #
  # @note   CONFIRM WHETHER STATE AND DEBUG STATE ARE NEEDED OR NOT.  MOST NOT CODED
  #         WHICH THEN INVOLVES PULLING FROM CLASS INTERNALS (MEMBER VARIABLES) WHICH
  #         MAY NOT BE KOSHER.
  #

  #============================== predict ==============================
  #
  #
  def predict(self):
    """!
    @brief  Predict next measurement, if applicable.

    Method overrides base, which does nothing.  This is because the measure
    function operates differently.

    @todo   Later on, there should be a code review and some unification of
            code and intent should be done. 
    """
    #TOREVIEW   - marker to find code review todo notes.

    self.detector.predict()
    if (self.filter is not None):
      self.filter.predict()

  #============================== measure ==============================
  #
  #
  def measure(self, I):
    """!
    @brief  Recover track point or track frame based on detector +
            trackPointer output.
   
    """

    ## First perform detection.  Due to operating assumptions, running
    ## the detector as is works just fine.
    #
    self.detector.measure(I)

    ## After that, get the detector state outcome and pass on to the
    ## trackpointer for processing.  For the puzzle piece perceiver,
    ## it can be a simple multi centroid track pointer or a puzzle board
    ## track pointer.  The appropriate filter should be defined in either
    ## case. Typical runs should use a puzzle board track pointer.
    #
    dState = self.detector.getState()
    self.tracker.process(dState.x)

    ## If there is a filter, additional processing occurs in the correction step.

  #============================== correct ==============================
  #
  #
  def correct(self):
    """!
    @brief  Correct the estimated state based on measured and predicted.

    At least if there is a filter defined.
    """
    if (self.filter is not None):
      trackOut = self.tracker.getOutput()
      self.filter.correct(trackOut)


  #=============================== adapt ===============================
  #
  #
  def adapt(self):
    """!
    @brief  Adapt parts of the process based on measurements and corrections.

    """

    # @note Not implemented. Deferring to when needed. For now, kicking to filter.
    # @note Should have config flag that engages or disengages, or member variable flag.
    if (self.filter is not None):
      self.filter.adapt()


  #============================= emptyState ============================
  #
  #
  def emptyState(self):
    """!
    @brief      Return state structure with no information.

    @param[out] estate  The state structure with no content.
    """

    # @todo     Should go review code that has this implemented and update.
    #           Argh.
    pass

  #============================== getState =============================
  #
  #
  def getState(self):
    """!
    @brief      Returns the current state structure.

    @param  cstate  The current state structure.
    """

    # @todo     Should go review code that has this implemented and update.
    #           Argh.
    pass


  # NOT IMPLEMENTED.  HERE AS REMINDER JUST IN CASE NEEDED IN FUTURE.
  #
  #============================= emptyDebug ============================
  #def emptyDebug(self):
  #============================== getDebug =============================
  #def getDebug(self):

  #============================= display_cv ============================
  #
  # @brief  Display any found track points on passed (color) image.
  #
  #
  def display_cv(self, I, ratio = None, window_name="puzzle pieces"):
    
    if (self.filter is None):

      if (self.tracker.haveMeas):
        display.trackpoints_cv(I, self.tracker.tpt, ratio, window_name)
      else:
        display.rgb_cv(I, ratio, window_name)

    else:

      not_done()

  #======================= buildWithBasicTracker =======================
  #
  # @todo   Should this be packaged up more fully with tracker config?
  #         Sticking to only detConfig is not cool since it neglects
  #         the tracker.
  #
  @staticmethod
  def buildWithBasicTracker(buildConfig):
    """!
    @brief  Given a stored detector configuration, build out a puzzle
            perceiver with multi-centroid tracking.

    Most of the configuration can default to standard settings or to
    hard coded puzzle settings (that should never be changed).
    """

    print(buildConfig)
    if (buildConfig.tracker is None):
      buildConfig.tracker = defaults.CfgCentMulti()

    if (buildConfig.perceiver is None):
      buildConfig.perceiver = perBase.CfgPerceiver()

    if (isinstance(buildConfig.detector, str)):
      matDetect    = Detector.load(buildConfig.detector)
    elif (isinstance(buildConfig.detector, CfgDetector)):
      matDetect    = Detector(buildConfig.detector)
    elif (buildConfig.detector is None):
      matDetect    = Detector()
    else:
      warnings.warn('Unrecognized black work mat detector configuration. Setting to default.')
      matDetect    = Detector()

    piecesTrack  = tracker.centroidMulti(None, buildConfig.tracker)
    piecesFilter = None

    perInst     = InstPuzzlePerceiver(detector = matDetect, 
                                      trackptr = piecesTrack,
                                      trackfilter = piecesFilter)

    return PuzzlePerceiver(buildConfig.perceiver, perInst)

#
#-------------------------------------------------------------------------
#=============================== Calibrator ==============================
#-------------------------------------------------------------------------
#

# @note Only loosely implemented. There is a calibrate2config method in the
#       Detector class that does the work. What's the right way to do it?
#

class Calibrator(detBlack.inCornerEstimator):

  # Definition below is what should be I think, but second one is as implemented.
  #def __init__(self, detCfg = None, processors=None, detModel = None):
  def __init__(self, processors=None, detModel = None):
    '''!
    @brief  Constructor for black workspace mat detector.

    @param[in]  processors  Image processors for the different layers.
    @param[in]  detModel    Detection models for the different layers.
    '''
    #@param[in]  detCfg      Detector configuration. NOT IMPLEMENTED.
    
    super(Calibrator,self).__init__(processors, detModel)
    self.mask = None


  #============================== setMask ==============================
  #
  def setMask(self, theMask):

    self.mask = theMask
    # @todo Should check dimensions.  For now ignoring.
    # TODO

  #================================ info ===============================
  #
  def info(self):
    #tinfo.name = mfilename;
    #tinfo.version = '0.1;';
    #tinfo.date = datestr(now,'yyyy/mm/dd');
    #tinfo.time = datestr(now,'HH:MM:SS');
    #tinfo.trackparms = bgp;
    pass

  #=============================== saveTo ==============================
  #
  def saveTo(self, fPtr):
    """!
    @brief  Save results of calibration to HDF5 file.

    The save process saves the necessary information to instantiate
    a Detector class object.  Stored in root of fPtr.

    @param[in] fPtr    An HDF5 file point.
    """

    super(Calibrator,self).saveTo(fPtr)

    if (self.mask is not None):
      fPtr.create_dataset("theMask", data=self.mask)

  # @todo   Why no load for this class?  Should include, otherwise it
  #         will be off.

  #----------------------------- saveConfig ----------------------------
  #
  # @todo   If saveTo save binary data + YAML configuration, do we really
  #         need a save config??  But not all are like that.  Need
  #         to work out a better system.  The YAML config should point to
  #         the necessary binary/HDF5 files to load for the initial state.
  #
  ## def saveConfig(self, outFile):
  ##   pass


#
#================================= BlackWorkMat ================================
