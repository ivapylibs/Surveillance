#================================= Glove =================================
'''!

@brief  Detector, track pointer, and perceiver classes for glove tracking.

Follows the structure of the Puzzle Scene perceiver, which packages everything
into one file since python has individual import facilities, and placing in one
uniform location simplifies things.  

Code here is copied from the Puzzle Scene glove tracker classes. The reason that
they were all mashed together in Puzzle Scene is to take advantage of common
image processing and not separate things such that efforts to reduce repeated
computation make the data passing too complex.  Changes should be mirrored
across these two files.

What should be contained in this file would be:
    1. Glove layer detector from RGBD input.
    2. Glove trackpointer based on layered detector output.
    3. Perceiver that combines detector + trackpointer.
    4. A calibration scheme for the entire process with saving to
        YAML and HDF5 files.

This single file replaces/supercedes the existing human_seg file in this 
directory (and by extension possibly others, like base_fg).
'''
#================================= Glove =================================

#
# @file Glove.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/29
#
#
# CODE NOTE:    Using 2 spaces for indent.
#               90 columns view. 8 space right margin.
#
#================================= Glove =================================

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
import ivapy.display_cv as display
from camera.base import ImageRGBD

from Surveillance.utils.region_grow import RG_Params
from Surveillance.utils.region_grow import MaskGrower

#--[0.C] PuzzleScene specific python libraries (ivapylibs)
#
from detector.Configuration import AlgConfig
from detector.inImage import fgImage
#import detector.bgmodel.inCorner as inCorner
import detector.fgmodel.Gaussian as Glove
from detector.base import DetectorState

import trackpointer.toplines as tglove
#import trackpointer.centroidMulti as tpieces

#import trackpointer.simple as simple

import perceiver.simple as perBase

#
#-------------------------------------------------------------------------
#============================= Configuration =============================
#-------------------------------------------------------------------------
#

class CfgGloveDetector(AlgConfig):
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
  
    init_dict = CfgGloveDetector.get_default_settings()
    super(CfgGloveDetector,self).__init__(init_dict, key_list, new_allowed)

    self.glove = Glove.CfgSGT(self.glove)

  #------------------------ get_default_settings -----------------------
  #
  @staticmethod
  def get_default_settings():

    fgGlove = Glove.CfgSGT.builtForRedGlove()
  
    default_settings = dict(workspace = dict(color = None, 
                                             mask  = None), 
                            glove = dict(fgGlove))
    
    return default_settings


#
#-------------------------------------------------------------------------
#============================ Setup Instances ============================
#-------------------------------------------------------------------------
#

@dataclass
class InstGloveDetector():
    '''!
    @brief Class for collecting visual processing methods needed by the
    PuzzleScene scene interpreter.

    '''
    workspace_mask  : np.ndarray
    glove : Glove.fgGaussian
 

#
#-------------------------------------------------------------------------
#================================ Detector ===============================
#-------------------------------------------------------------------------
#

@dataclass
class DetectorState:
  x         : any = None
  glove     : any = None


#============================== GloveByColor =============================
#
class GloveByColor(fgImage):
  """!
  @ingroup  Surveillance
  @brief    Glove detector by color only.
  """

  def __init__(self, detCfg = None, detInst = None, processors=None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  detInst     Detection instances for the different layers.
    @param[in]  processors  Image processors for the different layers.
    '''
    
    super(GloveByColor,self).__init__(processors)

    if (detCfg is None):
      detCfg = CfgGloveDetector()
      # @todo   This is wrong since it may not agree with detInst.
      #         Need to build from detInst if provided.

    self.mask = None

    if (detInst is not None):

      # @note   Commenting code for workspace color detector, not going to use.
      #self.workspace = detInst.workspace_color 
      self.glove     = detInst.glove 

      if (detInst.workspace_mask is not None):
        self.mask   = detInst.workspace_mask

    else:


      # @note   Workspace color detector ignored. Retaining just in case useful.
      #         Really reflects PuzzleScene code copy and downgrade with minimal
      #         code changes.  Should remove eventually if really not necessary.
      #         Just not yet certain how Glove tracking will be fully implemented.
      #self.workspace = None
      self.glove     = Glove.fgGaussian.buildFromCfg(detCfg.glove)

      # @note   Also probably useless.
      if (detCfg.workspace.mask is not None):
        self.mask   = detCfg.workspace.mask

    self.config   = detCfg

    self.imGlove  = None


  #------------------------------ predict ------------------------------
  #
  def predict(self):
    '''!
    @brief  Generate prediction of expected measurement.

    The detectors are mostly going to be static models, which means that
    prediction does nothing.  Just in case though, the prediction methods
    are called for them.
    '''

    #self.workspace.predict()
    self.glove.predict()

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   An RGB-D image (structure/dataclass).
    '''

    # @note Not dealing with pre-processor, but it might be important.
    # @todo Figure out how to use the improcessor.
    #
    #self.workspace.measure(I.color)
    #cDet = self.workspace.getState()    # Mask indicating what is puzzle mat.

    self.glove.measure(I)
    gDet = self.glove.getState()        # Mask indicating presumed glove region(s).

    image  = gDet.fgIm.astype('bool')

    if (self.config.glove.minArea > 0):
      morph.remove_small_objects(image, min_size = self.config.glove.minArea,
      connectivity = 1, out=image)

    #kernel  = np.ones((3,3), np.uint8)
    #moreGlove = scipy.ndimage.binary_dilation(image, kernel, 2)
    moreGlove = image
    nnz = np.count_nonzero(moreGlove)

    # @todo Need to clean up the code once finalized.
    if (nnz <= self.config.glove.minArea):
      moreGlove.fill(False)

    self.imGlove  = moreGlove.astype('bool')


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
    #self.workspace.adapt(onlyWS)
    #self.glove.correct(strictlyGlove)

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

    cState = DetectorState()

    gDet = self.glove.getState()
    cState.x   = 150*self.imGlove 
    cState.glove = self.imGlove

    return cState

  #----------------------------- emptyState ----------------------------
  #
  def emptyState(self):
    '''!
    @brief      Get and empty state to recover its basic structure.

    @param[out]  estate     The empty state.
    '''

    cState = DetectorState()
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
    self.glove.saveTo(fPtr)

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
    theDet = GloveByColor(theConfig)

  #================================ load ===============================
  #
  def load(inFile):
    fptr = h5py.File(inFile,"r")
    theDet = GloveByColor.loadFrom(fptr)
    fptr.close()
    return theDet

  #============================== loadFrom =============================
  #
  def loadFrom(fPtr):
    # Check if there is a mask

    fgGlove = Glove.fgGaussian.loadFrom(fPtr)

    keyList = list(fPtr.keys())
    if ("theMask" in keyList):
      print("Have a mask!")
      maskPtr = fPtr.get("theMask")
      wsMask  = np.array(maskPtr)
    else:
      wsMask  = None
      print("No mask.")

    detFuns = InstGloveDetector(workspace_mask  = wsMask,
                                glove           = fgGlove)

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

    #==[1]  Get the foreground color model.
    #
    print("\nThis step is for the (red) glove model; it is hard-coded.")
    if (initModel is None):
      fgModP  = Glove.SGMdebug(mu    = np.array([150.0,2.0,30.0]),
                               sigma = np.array([1100.0,250.0,250.0]) )

      fgModel = Glove.fgGaussian( Glove.CfgSGT.builtForRedGlove(), None, fgModP )
    else:
      print(initModel[0])
      print(initModel[1])
      fgModel = Glove.fgGaussian( initModel[0], None, initModel[1] )

    fgModel.refineFromStreamRGB(theStream, True)


    #==[2]  Package up and save as a configuration.  Involves instantiating a layered
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
    detFuns = InstGloveDetector(workspace_mask  = None,
                                glove           = fgModel)
    
    detPS = Detector(None, detFuns, None)
    detPS.save(outFile)



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
    
    self.glove  = tglove.fromBottom()

  #------------------------------ predict ------------------------------
  #
  def predict(self):
    '''!
    @brief  Generate prediction of expected measurement.

    The detectors are mostly going to be static models, which means that
    prediction does nothing.  Just in case though, the prediction methods
    are called for them.
    '''

    self.glove.predict()

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   Layered detection image instance (structure/dataclass).
    '''

    self.glove.measure(I.glove)

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

    self.glove.correct()

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
    
    if (self.glove.haveMeas):
      display.trackpoint(I, self.glove.tpt, ratio, window_name)
    else:
      display.rgb(I, ratio, window_name)



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
class InstGlovePerceiver():
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
    self.tracker.process(dState)

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

    gloveDet   = Detector.load(thefile)
    gloveTrack = TrackPointer()

    useMethods  = InstGlovePerceiver(detector=gloveDet, trackptr = gloveTrack, trackfilter = None)
    glovePerceiver = Perceiver(CfgExtra, useMethods)

    return glovePerceiver

#
#-------------------------------------------------------------------------
#=============================== Calibrator ==============================
#-------------------------------------------------------------------------
#

class CalibGloveByColor(GloveByColor):

  # @todo Need to flip: config, instances, processors. Align with super class.
  def __init__(self, detCfg = None, detInst = None, processors=None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  detInst     Detection instances for the different layers.
    @param[in]  processors  Image processors for the different layers.
    '''
    
    super(CalibGloveByColor,self).__init__(detCfg, processors, detModel)

    #self.workspace = detector.bgmodel.inCornerEstimator()
    #self.glove     = detector.fgmodel.fgGaussian()

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

    #self.workspace.predict()
    self.glove.predict()

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   An RGB-D image (structure/dataclass).
    '''

    #self.workspace.measure(I.color)
    self.glove.measure(I)

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
    #self.workspace.adapt(onlyWS)
    #self.glove.correct(strictlyGlove)

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
    self.glove.saveTo(fPtr)

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
#================================= Glove =================================
