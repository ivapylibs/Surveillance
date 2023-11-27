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

#import trackpointer.toplines as tglove
import trackpointer.centroidMulti as tpieces

#import trackpointer.simple as simple

import perceiver.simple as perBase

#
#-------------------------------------------------------------------------
#============================= Configuration =============================
#-------------------------------------------------------------------------
#

class CfgDetector(detBlack.CfgInCorner):
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
      init_dict = CfgDetector.get_default_settings()

    super(CfgDetector,self).__init__(init_dict, key_list, new_allowed)

  #------------------------ get_default_settings -----------------------
  #
  @staticmethod
  def get_default_settings():

    mat_settings = super(CfgDetector).get_default_settings()
    default_settings = dict(workspace = dict(color = mat_settings, 
                                             mask  = None)) 
    
    return default_settings

#
#-------------------------------------------------------------------------
#============================ Setup Instances ============================
#-------------------------------------------------------------------------
#

@dataclass
class InstBlackMat():
    '''!
    @brief Class for collecting visual processing methods needed by the
    Black Mat detection interpreter.

    '''
    workspace_color : detBlack.inCorner
    workspace_mask  : np.ndarray


#
#-------------------------------------------------------------------------
#================================ Detector ===============================
#-------------------------------------------------------------------------
#

class Detector(detBlack.inCorner):

  def __init__(self, detCfg = None, processors=None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  processors  Image processors for the different layers.
    '''

    super(Detectors,self).__init__(processors)

    if (detCfg is None):
      detCfg = CfgDetector()

    # @todo     THIS IS WRONG. FIX ONCE GET FLOW OF THINGS FIGURED OUT.
    self.workspace = detBlack.inCorner.buildFromCfg(detCfg.workspace.color)

    if (detCfg.workspace.mask is not None):
      self.mask   = detCfg.workspace.mask

    # @todo     SAME AS ABOVE.
    #self.imGlove  = None
    #self.imPuzzle = None


  #------------------------------ measure ------------------------------
  #
  def measure(self, I):

    super(Detector,self).measure(I)


  #DEBUG CODE --- IAMHERE

  #------------------------------ getState -----------------------------
  #
  def getState(self):
    '''!
    @brief      Get the complete detector state, which involves the 
                states of the individual layer detectors.

    @param[out]  state  The detector state for each layer, by layer.
    '''

    cState = DetectorsState()

    gDet = self.glove.getState()
    cState.x   = 150*self.imGlove + 75*self.imPuzzle 

    cState.glove = self.imGlove
    cState.pieces = self.imPuzzle

    return cState

  #----------------------------- emptyState ----------------------------
  #
  def emptyState(self):
    '''!
    @brief      Get and empty state to recover its basic structure.

    @param[out]  estate     The empty state.
    '''

    pass #for now. just getting skeleton code going.

  #------------------------------ getDebug -----------------------------
  #
  def getDebug(self):

    pass #for now. just getting skeleton code going.

  #----------------------------- emptyDebug ----------------------------
  #
  def emptyDebug(self):

    pass #for now. just getting skeleton code going.

  #-------------------------------- info -------------------------------
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
  #
  def saveTo(self, fPtr):    # Save given HDF5 pointer. Puts in root.
    '''!
    @brief     Save the instantiated Detector to given HDF5 file.

    The save process saves the necessary information to re-instantiate
    a Detectors class object. 

    @param[in] fPtr    An HDF5 file point.
    '''

    # Recursive saving to contained elements. They'll make their
    # own groups.
    self.workspace.saveTo(fPtr)
    self.depth.saveTo(fPtr)
    self.glove.saveTo(fPtr)

    fPtr.create_dataset("theMask", data=self.mask)

  #
  #-----------------------------------------------------------------------
  #============================ Static Methods ===========================
  #-----------------------------------------------------------------------
  #

  #---------------------------- buildFromCfg ---------------------------
  #
  @staticmethod
  def buildFromCfg(theConfig):
    '''!
    @brief  Instantiate from stored configuration file (YAML).
    '''
    theDet = Detectors(theConfig)

  #================================ load ===============================
  #
  def load(inFile):
    fptr = h5py.File(inFile,"r")
    theDet = Detectors.loadFrom(fptr)
    fptr.close()
    return theDet

  #============================== loadFrom =============================
  #
  def loadFrom(fPtr):
    # Check if there is a mask

    fgGlove = Glove.Gaussian.loadFrom(fPtr)
    wsColor = detBlack.inCorner.loadFrom(fPtr)
    wsDepth = onWorkspace.onWorkspace.loadFrom(fPtr)

    keyList = list(fPtr.keys())
    if ("theMask" in keyList):
      print("Have a mask!")
      maskPtr = fPtr.get("theMask")
      wsMask  = np.array(maskPtr)
    else:
      wsMask  = None
      print("No mask.")

    detFuns = InstPuzzleScene(workspace_color = wsColor,
                              workspace_depth = wsDepth,
                              workspace_mask  = wsMask,
                              glove           = fgGlove)

    detPS = Detectors(None, detFuns, None)
    return detPS
    

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
  def calibrate2config(theStream, outFile):

    #==[1]  Step 1 is to get the background color model.
    #       Hardcoded initial configuration with some refinement.
    #
    # @todo Should trace through code to see if this even does anything.
    #
    bgModel    = detBlack.inCorner.build_model_blackBG(-70, 0)
    bgDetector = Detector()

    bgDetector.set_model(bgModel)
    bgDetector.refineFromRGBDStream(theStream, True)

    #==[2]  Step 2 is to get the largest region of interest as a 
    #       workspace mask.  Then apply margins generated from refinement
    #       processing in the earlier step.
    #
    theMask = bgDetector.maskRegionFromRGBDStream(theStream, True)

    kernel  = np.ones((3,3), np.uint8)
    scipy.ndimage.binary_erosion(theMask, kernel, 2, output=theMask)

    bgDetector.apply_estimated_margins()
    bgDetector.bgModel.offsetThreshold(35)

    #
    # @todo Definitely can be improved.  Masking step and margin
    #       step can be combined.  Margin can be applied universally
    #       across image after averaging in mask region.  Offset
    #       threshold applied as needed.
    #

    #==[3]  Step 5 is to package up and save as a configuration.
    #       It involves instantiating a layered detector then
    #       saving the configuration.
    #   OR
    #       Manually saving as HDF5, possibly with YAML config string.
    #       Anything missing will need to be coded up.
    #       @todo   HDF5 save/load for YAML-based approaches (glove model).
    #       @todo   Maybe change up constructor for Detector.
    #       @todo   Add static member function to build out config from
    #               instances contained by layered detector.
    #               Approach is not fully settled and will take some
    #               baby step coding / modifications to get working.
    #
    detFuns = InstDetector(workspace_color = bgDetector,
                           workspace_mask  = theMask)
    
    detPS = Detectors(None, detFuns, None)
    detPS.save(outFile)


#
#-------------------------------------------------------------------------
#=============================== Perceiver ===============================
#-------------------------------------------------------------------------
#


# @todo Push to perceiver class??  Seems to be generic.
@dataclass
class InstPerceiver():
    '''!
    @brief Class for collecting visual processing methods needed by the
    PuzzleScene perceiver.

    '''
    detector : any
    trackptr : any
    trackfilter : any
    #to_update : any    # What role/purpose??

class Perceiver(perBase.simple):

  def __init__(self, perCfg = None, perInst = None):

    if perInst is not None:
      super().__init__(perCfg, perInst.detector, perInst.trackptr, perInst.trackfilter)
    else:
      raise Exception("Sorry, not yet coded up.") 
      # @todo   Presumably contains code to instantiate detector, trackptr, filter, etc.
    

  # @note   MOST OF THESE SHOULD BE THE SAME AS THE BASE PERCEIVER IMPLEMENTATION AND ARE NOT NEEDED.
  #         REVIEW AND DELETE THOSE THAT ARE NOT NEEDED.
  #
  # @note   CONFIRM WHETHER STATE AND DEBUG STATE ARE NEEDED OR NOT.  MOST NOT CODED WHICH THEN
  #         INVOLVES PULLING FROM CLASS INTERNALS (MEMBER VARIABLES) WHICH MAY NOT BE KOSHER.
  #

  def predict(self):
    self.detector.predict()
    if (self.filter is not None):
      self.filter.predict()

  def measure(self, I):
    # First perform detection.
    self.detector.measure(I)

    # Get state of detector. Pass on to trackpointer.
    dState = self.detector.getState()
    self.tracker.process(dState)

    # If there is a filter, get track state and pass on to filter.


  def correct(self):
    if (self.filter is not None):
      trackOut = self.tracker.getOutput()
      self.filter.correct(trackOut)


  def adapt(self):
    # @note Not implemented. Deferring to when needed. For now, kicking to filter.
    # @note Should have config flag that engages or disengages, or member variable flag.
    if (self.filter is not None):
      self.filter.adapt()

    pass

  def process(self, I):
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

    pass

  # IS this really needed??? Isn't it already done in measure?
  def detect(self):
    pass

  def getState(self):
    # What is this??
    pass

  def emptyState(self):
    pass

  def getDebugState(self):
    pass

  def emptyDebug(self):
    pass


#
#-------------------------------------------------------------------------
#=============================== Calibrator ==============================
#-------------------------------------------------------------------------
#

class Calibrator(Detectors):

  # @todo Need to flip: config, instances, processors. Align with super class.
  def __init__(self, detCfg = None, processors=None, detModel = None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  processors  Image processors for the different layers.
    @param[in]  detModel    Detection models for the different layers.
    '''
    
    super(Calibrator,self).__init__(processors)

    self.workspace = detector.bgmodel.inCornerEstimator()
    self.depth     = detector.bgmodel.onWorkspace()
    self.glove     = detector.fgmodel.Gaussian()

    self.phase     = None   # Need a phase enumerated type class.

    # Most likely need to do tiered or staged estimation.
    # Have the calibration or estimation process go through those
    # tiers/states.

  #------------------------------ predict ------------------------------
  #
  def predict(self):
    '''!
    @brief  Generate prediction of expected measurement.

    The detectors are mostly going to be static models, which means that
    prediction does nothing.  Just in case though, the prediction methods
    are called for them.
    '''

    self.workspace.predict()
    self.depth.predict()
    self.glove.predict()

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   An RGB-D image (structure/dataclass).
    '''

    self.workspace.measure(I.color)
    self.depth.measure(I.depth)
    self.glove.measure(I.color)

  #------------------------------ correct ------------------------------
  #
  def correct(self):
    '''!
    @brief  Apply correction process to the individual detectors.

    Apply naive correction on a per detector basis.  As a layered system,
    there might be interdependencies that would impact the correction step.
    Ignoring that for now since it does not immediately come to mind what
    needs to be done.  
    '''

    self.workspace.correct()
    self.depth.correct()
    self.glove.correct()

  #------------------------------- adapt -------------------------------
  #
  def adapt(self):
    '''!
    @brief  Adapt the layer detection models.

    This part is tricky as there may be dependencies across the layers
    in terms of what should be updated and what should not be.  Applying
    simple filtering to establish what pixels should adapt and which ones
    shouldn't.
    '''

    #--[1] Get the known background workspace layer, the known puzzle layer,
    #       the presumed glova layer, and the off workspace layer.  Use
    #       them to establish adaptation.

    #--[2] Apply adaption based on different layer elements.
    #
    #
    self.workspace.adapt(onlyWS)
    self.depth.adapt(offWS)
    self.glove.correct(strictlyGlove)

  #------------------------------ process ------------------------------
  #
  def process(self, I):
    '''!
    @brief      Apply entire predict to adapt process to source image.



    @param[in]  I   Source RGB-D image (structure/dataclass).
    '''

    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

  #------------------------------- detect ------------------------------
  #
  def detect(self, I):
    '''!
    @brief      Apply predict, measure, correct process to source image.

    Running detect alone elects not to adapt or update the underlying
    models.  The static model is presumed to be sufficient and applied
    to the RGBD stream.

    @param[in]  I   Source RGB-D image (structure/dataclass).
    '''

    self.predict()
    self.measure(I)
    self.correct()

  #------------------------------ getState -----------------------------
  #
  def getState(self):
    '''!
    @brief      Get the complete detector state, which involves the 
                states of the individual layer detectors.

    @param[out]  state  The detector state for each layer, by layer.
    '''

    pass #for now. just getting skeleton code going.

  #----------------------------- emptyState ----------------------------
  #
  def emptyState(self):
    '''!
    @brief      Get and empty state to recover its basic structure.

    @param[out]  estate     The empty state.
    '''

    pass #for now. just getting skeleton code going.

  #------------------------------ getDebug -----------------------------
  #
  def getDebug(self):

    pass #for now. just getting skeleton code going.

  #----------------------------- emptyDebug ----------------------------
  #
  def emptyDebug(self):

    pass #for now. just getting skeleton code going.

  #-------------------------------- info -------------------------------
  #
  def info(self):
    #tinfo.name = mfilename;
    #tinfo.version = '0.1;';
    #tinfo.date = datestr(now,'yyyy/mm/dd');
    #tinfo.time = datestr(now,'HH:MM:SS');
    #tinfo.trackparms = bgp;
    pass

  #-------------------------------- save -------------------------------
  #
  def saveTo(self, fPtr):

    self.workspace.saveTo(fPtr)
    self.glove.saveTo(fPtr)
    self.depth.saveTo(fPtr)

  #----------------------------- saveConfig ----------------------------
  #
  # @todo   If saveTo save binary data + YAML configuration, do we really
  #         need a save config??  But not all are like that.  Need
  #         to work out a better system.  The YAML config should point to
  #         the necessary binary/HDF5 files to load for the initial state.
  #
  def saveConfig(self, outFile):
    pass


#
#================================= BlackWorkMat ================================
