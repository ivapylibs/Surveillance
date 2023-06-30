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

from detector.Configuration import AlgConfig

import detector.inImageRGBD as detBase
import detector.bgmodel.inCorner as inCorner
import detector.bgmodel.onWorkspace as onWorkspace
import detector.fgmodel.Gaussian as Glove

from detector.inImageRGBD import inImageRGBD

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
    wsDepth = onWorkspace.CfgOnWS()
    fgGlove = Glove.CfgSGT.builtForRedGlove()
  
    default_settings = dict(workspace = dict(color = dict(wsColor), 
                                             depth = dict(wsDepth)), 
                            glove = dict(fgGlove))
    
    return default_settings


#
#-------------------------------------------------------------------------
#================================ Detector ===============================
#-------------------------------------------------------------------------
#


class Detectors(detBase.inImageRGBD):

  def __init__(self, detCfg = None, processors=None, detModel = None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  processors  Image processors for the different layers.
    @param[in]  detModel    Detection models for the different layers.
    '''
    
    super(inImageRGBD,self).__init__(processors)

    if (detCfg == None):
      detCfg = CfgPuzzleScene()

    self.workspace = inCorner.inCorner.buildFromCfg(detCfg.workspace.color)
    self.depth     = onWorkspace.onWorkspace.buildFromCfg(detCfg.workspace.depth)
    self.glove     = Glove.Gaussian.buildFromCfg(detCfg.glove)


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


  #---------------------------- buildFromCfg ---------------------------
  #
  @staticmethod
  def buildFromCfg(theConfig):
    '''!
    @brief  Instantiate from stored configuration file (YAML).
    '''
    theDet = Detectors(theConfig)

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
    
    super(inImageRGBD,self).__init__(processors)

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
