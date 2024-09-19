#================================= HoveringHand ================================
'''!

@brief  Detector, track pointer, and perceiver classes for hand tracking.

Follows the structure of the Puzzle Scene and Glove perceivers, which packages
everything into one module file since python has individual import facilities,
and placing in one uniform location simplifies things.  

Code here is copied from the Glove tracker classes but removes the color
components and presumes that only depth information is available for identifying
what is above a work surface. Since the color segmentation part is removed, this
method permits augmentation by a binary mask to provide more context for what
"hovering" pixels might be associated to a hand.  The reason being that depth is
not so precise and fingers/hand regions close to the surface do not register as
"hovering."  

The optional mask will operate sequentially and cannot exploit parallel
operation of things, at least if provided as input. This way a higher level
perceiver could run some things is pseudo-parallel.

What should be contained in this file would be:
    1. Hand layer detector from RGBD input.
    2. Hand trackpointer based on layered detector output.
    3. Perceiver that combines detector + trackpointer.
    4. A calibration scheme for the entire process with saving to
        YAML and HDF5 files.

@todo   Add separate option to apply the mask after detect, before track?
@todo   Did a bum rush through code to see if could finish up fast.  Needs a
        follow-up review and revision, especially as regards integration of
        hand segmentation/detection.
'''
#================================= HoveringHand ================================

#
# @file     HoveringHand.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/12/12
#
#
# CODE NOTE:    Using 2 spaces for indent.
#               90 columns view. 10 space right margin.
#
#================================= HoveringHand ================================

#--[0.A] Standard python libraries.
#
import numpy as np
import scipy
import cv2
from dataclasses import dataclass

import h5py

from skimage.segmentation import watershed
import skimage.morphology as morph

#--[0.B] custom python libraries (ivapylibs)
#
import camera.utils.display as display
from camera.base import ImageRGBD

from Surveillance.utils.region_grow import RG_Params
from Surveillance.utils.region_grow import MaskGrower

#--[0.C] PuzzleScene specific python libraries (ivapylibs)
#
from detector.Configuration import AlgConfig
import detector.inImageRGBD as detBase
import detector.bgmodel.onWorkspace as onWorkspace
#import detector.fgmodel.Gaussian as Glove
from detector.base import DetectorState

import trackpointer.toplines as thand
#import trackpointer.centroidMulti as tpieces

#import trackpointer.simple as simple

import perceiver.perceiver as perBase

#
#-------------------------------------------------------------------------
#============================= Configuration =============================
#-------------------------------------------------------------------------
#

class CfgHandDetector(AlgConfig):
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
  
    init_dict = CfgHandDetector.get_default_settings()
    super(CfgHandDetector,self).__init__(init_dict, key_list, new_allowed)

    self.workspace.depth = onWorkspace.CfgOnWS(self.workspace.depth)

  #------------------------ get_default_settings -----------------------
  #
  @staticmethod
  def get_default_settings():

    wsDepth = onWorkspace.CfgOnWS.builtForDepth435()
  
    default_settings = dict(workspace = dict(color = None, 
                                             depth = dict(wsDepth),
                                             mask  = None), 
                            minAreaHoles = 200,
                            minAreaHand  = 600,
                            hand = None)    # In case color version exists one day.
    
    return default_settings


#
#-------------------------------------------------------------------------
#============================ Setup Instances ============================
#-------------------------------------------------------------------------
#

@dataclass
class InstDetector():
    '''!
    @brief Class for collecting visual processing methods needed by the
    PuzzleScene scene interpreter.

    '''
    workspace_depth : onWorkspace.onWorkspace
    workspace_mask  : np.ndarray
    hand : any
 

#
#-------------------------------------------------------------------------
#================================ Detector ===============================
#-------------------------------------------------------------------------
#

@dataclass
class DetStateHand:
  x         : any = None
  hand      : any = None


class Detector(detBase.inImageRGBD):

  def __init__(self, detCfg = None, detInst = None, processors=None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  detInst     Detection instances for layer(s).
    @param[in]  processors  Image processors for layer(s).
    '''
    
    super(Detector,self).__init__(processors)

    if (detCfg is None):
      detCfg = CfgHandDetector()
      # @todo   This is wrong since it may not agree with detInst.
      #         Need to build from detInst if provided.

    self.mask = None

    if (detInst is not None):

      # @note   Commenting code for workspace color detector, not going to use.
      #self.workspace = detInst.workspace_color 
      self.depth    = detInst.workspace_depth
      self.hand     = detInst.hand 

      if (detInst.workspace_mask is not None):
        self.mask   = detInst.workspace_mask
        print("Have outlier mask!!")

    else:

      self.depth    = onWorkspace.onWorkspace.buildFromCfg(detCfg.workspace.depth)
      self.hand     = None

      # @note   Also probably useless.
      if (detCfg.workspace.mask is not None):
        self.mask   = detCfg.workspace.mask

    self.config  = detCfg
    self.imHand  = None


  #------------------------------ predict ------------------------------
  #
  def predict(self):
    '''!
    @brief  Generate prediction of expected measurement.

    The detectors are mostly going to be static models, which means that
    prediction does nothing.  Just in case though, the prediction methods
    are called for them.
    '''

    self.depth.predict()
    if (self.hand is not None):
      self.hand.predict()

  #------------------------------ measure ------------------------------
  #
  def measure(self, I, M = None):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   An RGB-D image (structure/dataclass).
    @param[in]  M   Optional mask indicate candidate hand regions (true) but
                    with presumption that there may false positives. 
    '''

    #==[1] Improcessing.
    #
    # @note Not dealing with pre-processor, but it might be important.
    # @todo Figure out how to use the improcessor.
    #

    #==[2] Generate hand mask(s).
    #
    self.depth.measure(I.depth)
    dDet = self.depth.getState()            # Mask indicating proximity to planar surface.

    if (self.hand is not None):             # Mask indicating potential hand region(s).
      self.hand.measure(I.color) 
      hDet = self.glove.getState()        

      if (M is None):                     
        M = hDet.fgIm
      else:
        np.logical_and(hDet.fgIm, M, out=M)

    #==[3] Process masks to generate final estimate of hand mask.
    #
    kernel      = np.ones((3,3), np.uint8)

    # @todo Redo so that there is an outlier mask and a region of interest mask.
    #       One uses OR operation and other uses AND, so they are different.
    #       If using outlier mask, then those pixels can never trigger a hand
    #       detection.  Hand should be big enough that this doesn't matter.
    #       Outliers should be in non-action regions or individual pixels.
    if (self.mask is not None):
      validMeas   = np.logical_or(self.mask, dDet.bgIm)
      nearSurface = scipy.ndimage.binary_dilation(validMeas, kernel, 3)  # Enlarge area
    else:
      nearSurface = scipy.ndimage.binary_dilation(dDet.bgIm, kernel, 3)  # Enlarge area

    # @note Above morphological processing enlarges area.  Earlier code in glove and
    #       in puzzle scene does opposite.  Should consider adjusting.
    #       Here it does a better job getting rid of noise.

    tooHigh     = np.logical_not(nearSurface)                           # Conservative.

    if M is None:
      defHand   = tooHigh
    else:
      defHand   = np.logical_and(M, tooHigh)

    if (self.config.minAreaHoles > 0):
      morph.remove_small_objects(defHand.astype('bool'), self.config.minAreaHoles, 2, out=defHand)

    # Commented code below is count based on mask (from M or combo with color mask).
    #lessHand = scipy.ndimage.binary_erosion(M, kernel, 5)
    #nnz      = np.count_nonzero(lessHand)
    #
    # Moving to count based on the defHand information.
    nnz = np.count_nonzero(defHand)

    # @todo Need to clean up the code once finalized.
    if (nnz > self.config.minAreaHand):
      if (M is not None):
        scipy.ndimage.binary_propagation(defHand, mask=M, structure=np.ones((3,3)),  output=defHand)
      # @note   Accidentally found scipy binary_propogation, which appears to
      #         support the desired hysteresis binary mask expansion.  If works,
      #         then should copy to other implementations.
      #         Have not tested because there is no external mask available.

      #DEBUG WHEN NNZ BIG ENOUGH.
      #display.binary_cv(moreHand, ratio=0.5, window_name="glove mask")

    else:
      defHand.fill(False)

    self.imHand  = defHand
    #DEBUG VISUALIZATION - EVERY LOOP
    #display.binary_cv(defHand,ratio=0.5,window_name="Hand")


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

    self.depth.correct()
    if (self.hand is not None):
      self.hand.correct()

  #------------------------------- adapt -------------------------------
  #
  def adapt(self):
    '''!
    @brief  Adapt the layer detection models.

    Does nothing.  There may nto be enough information to know how to
    proceed.
    '''
    # NOT IMPLEMENTED.
    #Pass through to individual adaptation routines.  How to manage is
    #not obvious.  Presume that default adaptation methods for workspace
    #depth and hand do the right thing.

    #--[1] Get the known background workspace layer, the known puzzle layer,
    #       the presumed glova layer, and the off workspace layer.  Use
    #       them to establish adaptation.

    #--[2] Apply adaption based on different layer elements.
    #
    #
    #self.depth.adapt(nearSurface)      # Regions known to be near surface. Not retained.
    #self.hand.correct(strictlyHand)   # Should get for sure glove regions.

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

    cState = DetStateHand()
    cState.hand = 150*self.imHand 
    cState.x    = self.imHand

    return cState

  #----------------------------- emptyState ----------------------------
  #
  def emptyState(self):
    '''!
    @brief      Get and empty state to recover its basic structure.

    @param[out]  estate     The empty state.
    '''

    cState = DetStateHand()
    return cState

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

    The save to function writes the necessary information to re-instantiate
    a Detectors class object to the passed HDF5 file pointer/instance. 

    @param[in] fPtr    An HDF5 file point.
    '''

    # Recursive saving to contained elements. They'll make their own groups.
    self.depth.saveTo(fPtr)
    if (self.hand is not None):
      self.hand.saveTo(fPtr)

    if (self.mask is not None):
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
    theDet = Detector(theConfig)

  #================================ load ===============================
  #
  def load(inFile):
    fptr = h5py.File(inFile,"r")
    theDet = Detector.loadFrom(fptr)
    fptr.close()
    return theDet

  #============================== loadFrom =============================
  #
  def loadFrom(fPtr):
    # Check if there is a mask

    #fgHand = Hand.Gaussian.loadFrom(fPtr)
    wsDepth = onWorkspace.onWorkspace.loadFrom(fPtr)

    keyList = list(fPtr.keys())
    if ("theMask" in keyList):
      print("Have a mask!")
      maskPtr = fPtr.get("theMask")
      wsMask  = np.array(maskPtr)
    else:
      wsMask  = None
      print("No mask.")

    detFuns = InstDetector(workspace_depth = wsDepth,
                           workspace_mask  = wsMask,
                           hand            = None) 
                                # @todo Figure out proper approach. TODO

    detPS = Detector(None, detFuns, None)
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
  # Since the detection schemes usually rely on an initial guess at the
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
  def calibrate2config(theStream, outFile, initModel = None):

    #==[1]  Get the depth workspace model.
    #
    print("\nThis step is for the depth model: count to 2 then quit.")
    theConfig = onWorkspace.CfgOnWS.builtForPuzzlebot()
    bgModel   = onWorkspace.onWorkspace.buildAndCalibrateFromConfig(theConfig, \
                                                                    theStream, True)

    #==[2]  Run background model for a bit and collect data that is consistently 
    #       true. In this case consistently is the median value.
    outMask = bgModel.estimateOutlierMaskRGBD(theStream, incVis = True, tauRatio = 0.9)

    kernel      = np.ones((3,3), np.uint8)
    scipy.ndimage.binary_dilation(outMask, kernel, 3, output=outMask)  # Shrink area

    display.binary_cv(outMask, window_name="Outlier Mask")
    print('Review mask and press key to continue')
    display.wait_cv()

    #==[3]  Get the foreground color model.
    #   ACTUALLY IGNORING FOR NOW SINCE NOT SURE HOW TO IMPLEMENT.
    #   IN PRINCIPLE CAN BE ANYTHING, WHICH MAKES IT DIFFICULT TO IMPLEMENT
    #   HERE.
    #
    # Do nothing.


    #==[4]  Package up and save as a configuration.  Involves instantiating a layered
    #       detector then saving the configuration.
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
    detFuns = InstDetector(workspace_depth = bgModel,
                           workspace_mask  = outMask,
                           hand            = None)
    
    detPS = Detector(None, detFuns, None)
    detPS.save(outFile)

    # CODE FROM LAYERED DETECTOR CONSTRUCTOR.  WILL BUILD ON OWN FROM
    # CONFIGURATION.  DOES NOT ACCEPT BUILT INSTANCES. ONLY OPTION IS
    # TO SAVE THEN LOAD UNLESS THIS CHANGES.
    #
    #self.depth     = onWorkspace.onWorkspace.buildFromCfg(detCfg.workspace.depth)
    #self.glove     = Hand.Gaussian.buildFromCfg(detCfg.glove)



#
#-------------------------------------------------------------------------
#============================== Trackpointer =============================
#-------------------------------------------------------------------------
#

class TrackPointer(object):

  def __init__(self, iState = None, trackCfg = None):
    '''!
    @brief  Constructor for layered puzzle scene tracker.

    @param[in]  iState      Initial state of tracks.
    @param[in]  trackCfg    Trackpointer(s) configuration.
    '''
    
    self.hand  = thand.fromBottom()

  #------------------------------ predict ------------------------------
  #
  def predict(self):
    '''!
    @brief  Generate prediction of expected measurement.

    The detectors are mostly going to be static models, which means that
    prediction does nothing.  Just in case though, the prediction methods
    are called for them.
    '''

    self.hand.predict()

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   Layered detection image instance (structure/dataclass).
    '''

    self.hand.measure(I)

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

    self.hand.correct()

  #------------------------------- adapt -------------------------------
  #
  def adapt(self):
    '''!
    @brief  Adapt the layer detection models.

    There is no adaptation. 
    '''
    pass

  #------------------------------ process ------------------------------
  #
  def process(self, I):
    '''!
    @brief      Apply entire predict to adapt process to source image(s).

    @param[in]  I   Layered detection image instance (structure/dataclass).
    '''
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

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

  #----------------------------- display_cv ----------------------------
  #
  # @brief  Display any found track points on passed (color) image.
  #
  #
  def display_cv(self, I, ratio = None, window_name="trackpoints"):
    
    if (self.hand.haveMeas):
      display.trackpoint_cv(I, self.hand.tpt, ratio, window_name)

    else:
      display.rgb_cv(I, ratio, window_name)



  #-------------------------------- info -------------------------------
  #
  def info(self):
    #tinfo.name = mfilename;
    #tinfo.version = '0.1;';
    #tinfo.date = datestr(now,'yyyy/mm/dd');
    #tinfo.time = datestr(now,'HH:MM:SS');
    #tinfo.trackparms = bgp;
    pass



#
#-------------------------------------------------------------------------
#=============================== Perceiver ===============================
#-------------------------------------------------------------------------
#
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

class Perceiver(perBase.Perceiver):

  #============================== __init__ =============================
  #
  def __init__(self, perCfg = None, perInst = None):

  
    print("Here I am!")

    if perInst is not None:
      super().__init__(perCfg, perInst.detector, perInst.trackptr, perInst.trackfilter)
    else:
      raise Exception("Sorry, not yet coded up.") 
      # @todo   Presumably contains code to instantiate detector, trackptr, 
      #         filter, etc. in the configuration.
    

  #=============================== predict ===============================
  #
  def predict(self):
    self.detector.predict()
    if (self.filter is not None):
      self.filter.predict()

  #=============================== measure ===============================
  #
  def measure(self, I):
    # First perform detection.
    self.detector.measure(I)

    # Get state of detector. Pass on to trackpointer.
    dState = self.detector.getState()
    self.tracker.process(dState.x)

    # If there is a filter, get track state and pass on to filter.


  #=============================== correct ===============================
  #
  def correct(self):
    if (self.filter is not None):
      trackOut = self.tracker.getOutput()
      self.filter.correct(trackOut)


  #================================ adapt ================================
  #
  def adapt(self):
    # @note Not implemented. Deferring to when needed. For now, kicking to filter.
    # @note Should have config flag that engages or disengages, or member variable flag.
    if (self.filter is not None):
      self.filter.adapt()

    pass

  #=============================== process ===============================
  #
  def process(self, I):
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

    pass

  #================================ detect ===============================
  #
  # IS this really needed??? Isn't it already done in measure?
  # I think the point here is that it is only a partial run through the
  # process and doesn't include the adaptation.  Such an implementation
  # doesn't make sense for a Perceiver since there is track pointing.
  # More like the top level should have a flag for adaptation on or off
  # of some kind of conditional adaptation parameter.
  #
  # @todo   Most likely needs to be removed.
  def detect(self):
    pass

  #=============================== getState ==============================
  #
  def getState(self):
    # What is this??
    pass

  #============================== emptyState =============================
  #
  def emptyState(self):
    pass

  #============================ getDebugState ============================
  #
  def getDebugState(self):
    pass

  #============================== emptyDebug =============================
  #
  def emptyDebug(self):
    pass

  #============================ buildFromFile ============================
  #
  # @todo   See what name should really be.
  # @todo   Just duplicated from perceive01glove.py.  Code may not be
  #         correct / final form.  Review overall code for consistency.
  #         Especially as relates to save/load process, in addition to name.
  #
  @staticmethod
  def buildFromFile(thefile, CfgExtra = None):

    handDet   = Detector.load(thefile)
    handTrack = TrackPointer()

    useMethods  = InstPerceiver(detector=handDet, trackptr = handTrack, trackfilter = None)
    handPerceiver = Perceiver(CfgExtra, useMethods)

    return handPerceiver

#
#-------------------------------------------------------------------------
#=============================== Calibrator ==============================
#-------------------------------------------------------------------------
#

class Calibrator(Detector):

  # @todo Need to flip: config, instances, processors. Align with super class.
  def __init__(self, detCfg = None, processors=None, detModel = None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  processors  Image processors for the different layers.
    @param[in]  detModel    Detection models for the different layers.
    '''
    
    super(Calibrator,self).__init__(processors)

    #self.workspace = detector.bgmodel.inCornerEstimator()
    self.depth     = detector.bgmodel.onWorkspace()
    self.hand      = None #detector.fgmodel.Gaussian()

    self.phase     = None   # Need a phase enumerated type class.

    # Most likely need to do tiered or staged estimation.
    # Have the calibration or estimation process go through those
    # tiers/states.

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   An RGB-D image (structure/dataclass).
    '''

    #self.workspace.measure(I.color)

    # @todo How to order and how to use is not obvious.  Need to fix.
    #       In particular, need to establish exactly how the hand
    #       detector should operate and in what order.  The depth
    #       code assumes a mask is available.  Perhaps hand should go
    #       first, followed by depth.
    #
    if (self.hand is not None):
      self.hand.measure(I.color)

      # @todo Figure out how to bind. For now not bound. Needs to change.
      #handMask = self.hand.getMask()
      #self.depth.measure(I.depth, handMask)
      self.depth.measure(I.depth)

    else:
      self.depth.measure(I.depth)

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

    #self.workspace.correct()
    self.depth.correct()
    if (self.hand is not None):
      self.hand.correct()

  #------------------------------- adapt -------------------------------
  #
  def adapt(self):
    '''!
    @brief  Adapt the layer detection models.

    Doing nothing.  Need to review.

    @todo   Figure out proper approach here.
    '''

    #--[1] Get the known background workspace layer, the known puzzle layer,
    #       the presumed glova layer, and the off workspace layer.  Use
    #       them to establish adaptation.

    #--[2] Apply adaption based on different layer elements.
    #
    #
    #self.workspace.adapt(onlyWS)
    #self.depth.adapt(offWS)
    #self.glove.correct(strictlyHand)

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

    #self.workspace.saveTo(fPtr)
    self.depth.saveTo(fPtr)

    # @todo This is bad idea.  Should put in hand sub-folder.
    if (self.hand is not None):
      self.hand.saveTo(fPtr)

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
#================================= HoveringHand ================================
