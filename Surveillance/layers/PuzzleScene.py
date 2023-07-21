#============================== PuzzleScene ==============================
'''!

@brief  Layered detector, track pointer, and perceiver classes for puzzle
        surveillance system.


Given how python works regarding code import, it seems like the best would
be to simply put all of the code into this one file.  That will make it 
rather long, but with a consistent coding interface.  If done properly,
the individual classes will lean heavily on other code libraries and be
relatively compact.

What should be contained in this file would be:
    1. Layered detector from RGBD input.
    2. Layered trackpointers based on layered detector output.
    3. Layered perceiver that combine detector + trackpointers.
    4. A calibration scheme for the entire process with saving to
        YAML and HDF5 files.

This single file replaces/supercedes the existing files in this directory
(human_seg, robot_seg, tabletop_seg, puzzle_seg, base_bg, base_fg, base).
'''
#============================== PuzzleScene ==============================

#
# @file PuzzleScene.py
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/29
#
#============================== PuzzleScene ==============================

import numpy as np
import scipy
import cv2

import camera.utils.display as display
from camera.base import ImageRGBD

from skimage.segmentation import watershed

from Surveillance.utils.region_grow import RG_Params
from Surveillance.utils.region_grow import MaskGrower

from detector.Configuration import AlgConfig

import detector.inImageRGBD as detBase
import detector.bgmodel.inCorner as inCorner
import detector.bgmodel.onWorkspace as onWorkspace
import detector.fgmodel.Gaussian as Glove
from detector.inImage import detectorState


#import trackpointer.simple as simple

import perceiver.simple as perBase

#
#-------------------------------------------------------------------------
#============================= Configuration =============================
#-------------------------------------------------------------------------
#

class CfgPuzzleScene(AlgConfig):
  '''!
  @brief    Configuration instance for Puzzle Scene perceiver.  Designed
            to work for processing subsets (detect, track, etc).
  '''
  #------------------------------ __init__ -----------------------------
  #
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):
    '''!
    @brief    Instantiate a puzzle scene configuration object.
  
    '''
  
    init_dict = CfgPuzzleScene.get_default_settings()
    super(CfgPuzzleScene,self).__init__(init_dict, key_list, new_allowed)

    self.workspace.color = inCorner.CfgInCorner(self.workspace.color)
    self.workspace.depth = onWorkspace.CfgOnWS(self.workspace.depth)
    self.glove = Glove.CfgSGT(self.glove)

  #------------------------ get_default_settings -----------------------
  #
  @staticmethod
  def get_default_settings():

    wsColor = inCorner.CfgInCorner()
    wsDepth = onWorkspace.CfgOnWS.builtForDepth435()
    fgGlove = Glove.CfgSGT.builtForRedGlove()
  
    default_settings = dict(workspace = dict(color = dict(wsColor), 
                                             depth = dict(wsDepth),
                                             mask  = None), 
                            glove = dict(fgGlove))
    
    return default_settings


#
#-------------------------------------------------------------------------
#================================ Detector ===============================
#-------------------------------------------------------------------------
#


class Detectors(detBase.inImageRGBD):

  def __init__(self, detCfg = None, processors=None, detInst = None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  processors  Image processors for the different layers.
    @param[in]  detInst     Detection instances for the different layers.
    '''
    
    super(Detectors,self).__init__(processors)

    if (detInst is not None):

      self.workspace = detInst.workspace.color 
      self.depth     = detInst.workspace.depth
      self.glove     = detInst.glove 

      if (detCfg is not None) and (detCfg.workspace.mask is not None):
        self.mask   = detCfg.workspace.mask

    else:

      if (detCfg is None):
        detCfg = CfgPuzzleScene()

      self.workspace = inCorner.inCorner.buildFromCfg(detCfg.workspace.color)
      self.depth     = onWorkspace.onWorkspace.buildFromCfg(detCfg.workspace.depth)
      self.glove     = Glove.Gaussian.buildFromCfg(detCfg.glove)

      if (detCfg.workspace.mask is not None):
        self.mask   = detCfg.workspace.mask

    self.imGlove  = None
    self.imPuzzle = None


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

    # @note Not dealing with pre-processor, but it might be important.
    # @todo Figure out how to use the improcessor.
    #
    self.workspace.measure(I.color)
    self.depth.measure(I.depth)
    self.glove.measure(I.color)

    cDet = self.workspace.getState()
    dDet = self.depth.getState()
    gDet = self.glove.getState()

    tooHigh    = np.logical_not(dDet.bgIm)
    defGlove   = np.logical_and(gDet.fgIm, tooHigh)
    notBoard   = np.logical_and(np.logical_not(cDet.x), dDet.bgIm)

    marker = np.add(defGlove.astype('uint32'), dDet.bgIm.astype('uint32'))
    image  = gDet.fgIm.astype('uint8')

    kernel = np.ones((3,3), np.uint8)
    lessGlove = scipy.ndimage.binary_erosion(gDet.fgIm, kernel, 5)
    nnz = np.count_nonzero(lessGlove)

    if (nnz > 100):
      if (False):
        moreGlove = scipy.ndimage.binary_dilation(gDet.fgIm, kernel, 1)
        image = 50*np.logical_not(moreGlove).astype('uint8') 
        defGlove = watershed(image, np.logical_and(defGlove,lessGlove), mask=moreGlove)
      else:
        moreGlove = scipy.ndimage.binary_dilation(lessGlove, kernel, 3)
        np.logical_and(defGlove, moreGlove, out=defGlove)
     
      #DEBUG
      #display.gray_cv(20*image, ratio=0.5, window_name="WSimage")
      #print(np.shape(wsout))
      #print(type(wsout))
      #display.binary_cv(wsout, ratio=0.5, window_name="WSlabel")
      pass

    else:
      defGlove.fill(False)

    self.imGlove  = defGlove.astype('bool')
    self.imPuzzle = notBoard


    #ATTEMPT 1: Using OpenCV watershed
    #  wsout  = watershed(image, marker)
    #  display.gray_cv(100*gm.astype('uint8'), ratio=0.5, window_name="WS")
    #
    # Abandoned because of silly OpenCV type issues/conflicts.

    # ATTEMPT 2: Using scikit.image watershed
    #
    #  Figured out an ugly way to make it work.  Need to fix later.
    #  Maybe in C++ re-implementation.  Works well enough for now.
    #  Review after better detector calibration.
    #
    #   tooHigh    = np.logical_not(dDet.bgIm)
    #   defGlove   = np.logical_and(gDet.fgIm, tooHigh)
    #   notBoard   = np.logical_and(np.logical_not(cDet.x), dDet.bgIm)
    #
    #   image =  10*cDet.x.astype('uint8') + 10*tooHigh.astype('uint8') - 2*gDet.fgIm.astype('uint8') + 1*defGlove.astype('uint8')
    #
    #   wsout = watershed(image, defGlove, mask=np.logical_not(cDet.x))
    #   display.gray_cv(20*image, ratio=0.5, window_name="WSimage")
    #   print(np.shape(wsout))
    #   print(type(wsout))
    #   display.binary_cv(wsout, ratio=0.5, window_name="WSlabel")
    #
    # Really dumb interface.  Having trouble implementing.  The documentation
    # is quite poor on this front.  I thought it would only go up, but
    # apparently it can also go down.  Annoying.  Ends up oversegmenting.
    # Looks like it will be difficult to control.
    #
    # Right now, first see if can clean up binary masks with the stored
    # mask region.
    #
    # 

    # ATTEMPT 3: Using scikit.image flood fill.
    #
    #    wnz = np.argwhere(defGlove)    # Use instead of nnz. nnz = #rows of wnz.
    #
    #    gcent = np.fix(np.mean(wnz,0))
    #    gcent = gcent.astype('int')
    #    gcent = tuple(gcent)
    #    gloveNot = np.logical_not(gDet.fgIm)
    #    gm = flood(gloveNot, gcent)
    #
    # Abandoned because takes only a single seed point and won't grow outwards.
    # There is no good way to pick the seed point. Centroid sometimes fails.
    # Picking one randomly might lead to an off glove choice as there appear
    # to be random misclassified points.  Maybe should be fixed.

    # ATTEMPT 3: ABSURDLY SLOW. WORKS BUT HIGHLY NOT RECOMMENDED.
    #   Yiye's region grower written in python. Really bad idea.
    #   Even slower than a Matlab based implementation.
    #
    #  region_grower = MaskGrower(RG_Params)
    #  region_grower.process_mask(gDet.fgIm,defGlove)
    #
    #  gm = region_grower.final_mask
    #
    # Right idea but abandoned due to speed issues.  Might be better to just
    # code up in C and write python wrapper. Overall, probably best to kick
    # this python can down the road.
    #
    # Amazing to realize that something of this sort doesn't exist as part
    # of any image processing toolboxes. Even Matlab supports this!
    #

    # ATTEMPT 4: Use an aggressive erosion then apply glove mask.
      # OpenCV: not good. stupid type issue most likely. 
      # python is annoying.
      # display.binary_cv(defGlove, ratio=0.5, window_name="oldGlove")
      #
      #kernel = np.ones((5,5), np.uint8)
      #defGlove = cv2.erode(defGlove.astype('uint8'), kernel, 3)
      #
      # scipy: no good.  requires too much dilation.
      # probably best to go back to watershed and be aggressive.
      #kernel = np.ones((5,5), np.uint8)
      #defGlove = scipy.ndimage.binary_dilation(defGlove.astype('uint8'), kernel, 10)
      #defGlove = np.logical_and(defGlove, gDet.fgIm)
      #display.binary_cv(defGlove, ratio=0.5, window_name="newGlove")
      



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

    cState = detectorState()

    gDet = self.glove.getState()
    cState.x   = 150*self.imGlove + 75*self.imPuzzle 

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


  #---------------------------- buildFromCfg ---------------------------
  #
  @staticmethod
  def buildFromCfg(theConfig):
    '''!
    @brief  Instantiate from stored configuration file (YAML).
    '''
    theDet = Detectors(theConfig)


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
    bgModel    = inCorner.inCorner.build_model_blackBG(-70, 0)
    bgDetector = inCorner.inCornerEstimator()

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

    #==[3]  Step 4 is to get the depth workspace model.
    #
    theConfig = onWorkspace.CfgOnWS.builtForDepth435()
    bgModel   = onWorkspace.onWorkspace.buildAndCalibrateFromConfig(theConfig, \
                                                                    theStream, True)

    #==[4]  Step 3 is to get the foreground color model.
    #
    fgModP  = Glove.SGMdebug(mu    = np.array([150.0,2.0,30.0]),
                           sigma = np.array([1100.0,250.0,250.0]) )
    fgModel = Glove.Gaussian( Glove.CfgSGT.builtForRedGlove(), None, fgModP )

    fgModel.refineFromRGBDStream(theStream, True)


    #==[5]  Step 5 is to package up and save as a configuration.
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

    # CODE FROM LAYERED DETECTOR CONSTRUCTOR.  WILL BUILD ON OWN FROM
    # CONFIGURATION.  DOES NOT ACCEPT BUILT INSTANCES. ONLY OPTION IS
    # TO SAVE THEN LOAD UNLESS THIS CHANGES.
    #
    #self.workspace = inCorner.inCorner.buildFromCfg(detCfg.workspace.color)
    #self.depth     = onWorkspace.onWorkspace.buildFromCfg(detCfg.workspace.depth)
    #self.glove     = Glove.Gaussian.buildFromCfg(detCfg.glove)



#
#-------------------------------------------------------------------------
#============================= Trackpointers =============================
#-------------------------------------------------------------------------
#

class Trackpointers(object):

  def __init__(self, iState = None, trackCfg = None):
    '''!
    @brief  Constructor for layered puzzle scene tracker.

    @param[in]  iState      Initial state of tracks.
    @param[in]  trackCfg    Trackpointer(s) configuration.
    '''
    
    # Will most likely need to differentiate in play vs placed pieces.

    #self.piecesInPlay = trackpointer.centroidMulti
    #self.piecesPlaced = trackpointer.centroidMulti
    self.pieces = trackpointer.centroidMulti
    self.glove  = tackpointer.top19

  #------------------------------ predict ------------------------------
  #
  def predict(self):
    '''!
    @brief  Generate prediction of expected measurement.

    The detectors are mostly going to be static models, which means that
    prediction does nothing.  Just in case though, the prediction methods
    are called for them.
    '''

    self.pieces.predict()
    self.glove.predict()

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   Layered detection image instance (structure/dataclass).
    '''

    self.glove.measure(I.glove)
    self.pieces.measure(I.pieces)

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
    self.pieces.correct()

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

class Perceiver(perBase.simple):

  def __init__(self):
    pass

  def predict(self):
    pass

  def measure(self):
    pass

  def correct(self):
    pass

  def adapt(self):
    pass

  def process(self):
    pass

  def detect(self):
    pass

  def getState(self):
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
#============================== PuzzleScene ==============================
