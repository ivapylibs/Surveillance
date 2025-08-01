#============================== PuzzleScene ==============================
##
# @package  Surveillance.layers.PuzzleScene
#
# @brief    Layered detector, track pointer, and perceiver classes for puzzle
#           surveillance system.
#
#
# Given how python works regarding code import, it seems like the best would be
# to simply put all of the code into this one file.  That will make it rather
# long, but with a consistent coding interface.  If done properly, the
# individual classes will lean heavily on other code libraries and be
# relatively compact.
# 
# What should be contained in this file would be:
#     1. Layered detector from RGBD input.
#     2. Layered trackpointers based on layered detector output.
#     3. Layered perceiver that combine detector + trackpointers.
#     4. A calibration scheme for the entire process with saving to
#         YAML and HDF5 files.
# 
# This single file replaces/supercedes the existing files in this directory
# (human_seg, robot_seg, tabletop_seg, puzzle_seg, base_bg, base_fg, base).
#
# @ingroup  Surveillance
#
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2023/06/29
#
#============================== PuzzleScene ==============================

#--[0.A] Standard python libraries.
#
import numpy as np
import scipy
import cv2
from dataclasses import dataclass
from collections import Counter

import h5py

from skimage.segmentation import watershed

#--[0.B] custom python libraries (ivapy)
#
import ivapy.display_cv as display
from camera.base import ImageRGBD

from Surveillance.utils.region_grow import RG_Params
from Surveillance.utils.region_grow import MaskGrower

#--[0.C] PuzzleScene specific python libraries (ivapylibs)
#
from detector.Configuration import AlgConfig
import detector.inImageRGBD as detBase
import detector.bgmodel.inCorner as inCorner
import detector.bgmodel.onWorkspace as onWorkspace
import detector.fgmodel.Gaussian as Glove
#from detector.base import DetectorState

import trackpointer.toplines as tglove
import trackpointer.centroidMulti as tpieces


#import trackpointer.simple as simple

from perceiver.perceiver import Perceiver 
from perceiver.perceiver import PerceiverState
from perceiver.monitor   import Monitor
from detector.activity.byRegion import imageRegions 


#
#-------------------------------------------------------------------------------
#================================ Configuration ================================
#-------------------------------------------------------------------------------
#

class CfgPuzzleScene(AlgConfig):
  '''!
  @ingroup  Surveillance
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
    self.pieces = tpieces.CfgCentMulti(self.pieces)

  #------------------------ get_default_settings -----------------------
  #
  @staticmethod
  def get_default_settings():

    wsColor  = inCorner.CfgInCorner()
    wsDepth  = onWorkspace.CfgOnWS.builtForDepth435()
    fgGlove  = Glove.CfgSGT.builtForRedGlove()
    trackPcs = tpieces.CfgCentMulti()
    trackPcs.minArea = 100
  
    default_settings = dict(workspace = dict(color = dict(wsColor), 
                                             depth = dict(wsDepth),
                                             mask  = None), 
                            glove  = dict(fgGlove),
                            pieces = dict(trackPcs))
    
    return default_settings

#
#-------------------------------------------------------------------------
#============================ Setup Instances ============================
#-------------------------------------------------------------------------
#

@dataclass
class InstPuzzleScene():
    '''!
    @ingroup    Surveillance
    @brief      Class for collecting visual processing methods needed by the
                puzzle scene interpreter.
    '''
    workspace_color : inCorner.inCorner
    workspace_depth : onWorkspace.onWorkspace
    workspace_mask  : np.ndarray
    glove : Glove.fgGaussian
 

#
#-------------------------------------------------------------------------------
#=============================== PuzzleDetectors ===============================
#-------------------------------------------------------------------------------
#

@dataclass
class StateDetectors:
  '''!
  @ingroup  Surveillance
  @brief    Puzzle detector state information.

  The state information consists of:
  field  | contents
  ------ | --------
  x      | Combined grayscale image hand/glove + pieces.
  hand   | Binary mask of hand/glove region.
  pieces | Binary mask of puzzle pieces regions.
  isHand | Boolean indicating that hand is in scene.
  '''
  x         : any = None        
  hand      : any = None        
  pieces    : any = None        
  isHand    : bool = False      


class PuzzleDetectors(detBase.inImageRGBD):
  '''!
  @ingroup  Surveillance
  @brief    Detector for layered puzzle scene: glove and puzzle pieces. 

  Puzzle pieces are really flat, non-background elements on the work mat.
  Anything high enough off the work mat is not a puzzle piece, but presumed to
  be a hand or other equivalent puzzle manipulation mechanism.
  '''

  def __init__(self, detCfg = None, detInst = None, processors=None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration (from CfgPuzzleScene).
    @param[in]  processors  Image processors for the different layers.
    @param[in]  detInst     Detection instances for the different layers.
    '''
    
    super(PuzzleDetectors,self).__init__(processors)

    if (detInst is not None):

      self.workspace = detInst.workspace_color 
      self.depth     = detInst.workspace_depth
      self.glove     = detInst.glove 

      if (detInst.workspace_mask is not None):
        self.mask   = detInst.workspace_mask

    else:

      if (detCfg is None):
        detCfg = CfgPuzzleScene()

      self.workspace = inCorner.inCorner.buildFromCfg(detCfg.workspace.color)
      self.depth     = onWorkspace.onWorkspace.buildFromCfg(detCfg.workspace.depth)
      self.glove     = Glove.fgGaussian.buildFromCfg(detCfg.glove)

      if (detCfg.workspace.mask is not None):
        self.mask   = detCfg.workspace.mask

    self.imGlove  = None
    self.imPuzzle = None
    self.hand     = False

    self.params   = detCfg


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

    ##  First, perform any specified pre-processing.
    #
    # @note Not dealing with pre-processor, but it might be important.
    # @todo Figure out how to use the improcessor.
    #

    ##  Second, invoke the layer detectors and post-processor to differentiate
    ##  the actual semantic layers of the scene.  The layer detectors should be
    ##  considered as raw detectors that need further polishing to extract the
    ##  desired semantic layer information.  These layers are further processed
    ##  by customized track pointers and filters.
    #
    self.workspace.measure(I.color)
    self.depth.measure(I.depth)
    self.glove.measure(I.color)

    cDet = self.workspace.getState()    # Binary mask region of puzzle mat.
    dDet = self.depth.getState()        # Binary mask region close to planar surface.
    gDet = self.glove.getState()        # Binary mask region of presumed glove.

    ##  The post processing here is hard-coded rather than a private member function
    ##  invocation.  
    #
    #   Shrink region associated to the surface just to clean up potential sources of
    #   confusion and promote recovery of puzzle pieces that are fully captured.
    kernel = np.ones((3,3), np.uint8)
    nearSurface = scipy.ndimage.binary_erosion(dDet.bgIm, kernel, 3)

    #
    #   The glove regions should be above the surface and pass the glove detector.
    #   Otherwise, they could be confounding puzzle pieces that look like the glove.
    #   Anything near the surface and not the black mat is most likely a puzzle piece.
    #   Capture these, as they can be interpreted as regions not being the black mat
    #   surface but level with the surface.  Note that "level with the surface" is
    #   only accurate up to the depth camera's depth sensitivity.  Some can be a
    #   little too noisy to really capture fine details. 
    #
    tooHigh          = np.logical_not(nearSurface)
    count = np.count_nonzero(tooHigh)
    # Hard coded temporarily
    self.hand = (count > 30000)

    defGlove         = np.logical_and(gDet.fgIm, tooHigh)
    SurfaceButNotMat = np.logical_and(np.logical_not(cDet.x), nearSurface)

    # @todo marker and image not used. Part of older code.  Should remove once
    #       actual implementation locked down and still not used.
    # TODO
    #marker = np.add(defGlove.astype('uint32'), dDet.bgIm.astype('uint32'))
    #image  = gDet.fgIm.astype('uint8')

    #   Code below is an attempt to employ detection hysteresis for the glove.
    #   A much smaller set is established as the definitely glove seed region.
    #   That seed region should subsequently get expanded (later code) into a
    #   bigger region based on connectivity.  It has been very troublesome to
    #   find that sort of operation in existing libraries.  Many attempts were
    #   made but they all do the wrong thing.  Amazing that such a feaure does
    #   not exist. It's a core component of standard detection focused image
    #   processing strategies.  So silly.
    #
    #lessGlove = scipy.ndimage.binary_erosion(gDet.fgIm, kernel, 5)
        # NOTE  2023/12/06  Trying out something new in line below vs line above.
    lessGlove = scipy.ndimage.binary_erosion(defGlove, kernel, 5)
    moreGlove = scipy.ndimage.binary_dilation(defGlove, kernel, 5)
    nnz = np.count_nonzero(lessGlove)

    # @todo Make the nnz count threshold a parameter. For now hard coded.
    #       Check if already done in the YAML file.
    #TODO
    if (nnz > 100):
      if (False):
        # Failed attempt at hysteresis.
        moreGlove = scipy.ndimage.binary_dilation(gDet.fgIm, kernel, 1)
        image = 50*np.logical_not(moreGlove).astype('uint8') 
        defGlove = watershed(image, np.logical_and(defGlove,lessGlove), mask=moreGlove)
      else:
        #moreGlove = scipy.ndimage.binary_dilation(lessGlove, kernel, 3)
        #np.logical_and(defGlove, moreGlove, out=defGlove)
        tipPt   = tglove.tipFromBottom(lessGlove)
        startIm = lessGlove.astype('uint8') 
        #mask1 = cv2.copyMakeBorder(cDet.x.astype('uint8'), 1, 1, 1, 1, cv2.BORDER_CONSTANT, 1)
        # NOTE  2023/12/06  Trying out something new in line below vs line above.
        mask1 = cv2.copyMakeBorder(moreGlove.astype('uint8'), 1, 1, 1, 1, \
                                                              cv2.BORDER_CONSTANT, 1)
        _,defGlove,_,_ = cv2.floodFill(startIm, mask1,  \
                                       (int(tipPt[0]),int(tipPt[1])), 1, 1, 1)
        # Grow out into non background color regions.  Snags nearby puzzle pieces too.
        # Seems like a feature and not a bug. Allows for them to be ignored as occluded.
        # @note Current hysteresis binary mask expansion does not work.  Need to code own.
        # @note Made changes on 12/06.  Need to confirm functionality.
        #TODO TOTEST

      #DEBUG VISUALS WHEN NNZ BIG ENOUGH.
      #display.gray(20*image, ratio=0.5, window_name="WSimage")
      #print(np.shape(wsout))
      #print(type(wsout))
      #display.binary(wsout, ratio=0.5, window_name="WSlabel")

    else:
      # Zero out glove regions. It is not present or not in active area (moving out of
      # the field of view, or just entered but not quite fully in).
      defGlove.fill(False)

    ##  Package the processed layers started with the glove.  Next, remove any
    ##  parts of the not surface layer that intersect with the expanded glove 
    ##  region.  May remove adjacent puzzle piece area; that's OK since we can't
    ##  rely on those pieces having been fully measured/captured.
    ##  After that
    #
    self.imGlove = defGlove.astype('bool')
    SurfaceButNotMat   = np.logical_and(SurfaceButNotMat, np.logical_not(moreGlove))

    if (self.mask is not None):
      self.imPuzzle = np.logical_and(SurfaceButNotMat, self.mask)
    else:
      self.imPuzzle = SurfaceButNotMat

    #DEBUG VISUALIZATION - EVERY LOOP
    #display.binary(dDet.bgIm,window_name="too high")


    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #
    # CODE BELOW DOCUMENTS DIFFERENT APPROACHES TO CAPTURING THE GLOVE.
    # MANY OF THEM FAILED BECAUSE THE API DOESN'T SUPPORT THE DESIRED IMAGE 
    # PROCESSING EVEN THOUGH IT SHOULD.  THE UNDERLYING IMPLEMENTATION DETAILS
    # DON'T SUPPORT IT IN SPITE OF THE OPERATION SEEMING TO.
    #
    #ATTEMPT 1: Using OpenCV watershed
    #  wsout  = watershed(image, marker)
    #  display.gray(100*gm.astype('uint8'), ratio=0.5, window_name="WS")
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
    #   display.gray(20*image, ratio=0.5, window_name="WSimage")
    #   print(np.shape(wsout))
    #   print(type(wsout))
    #   display.binary(wsout, ratio=0.5, window_name="WSlabel")
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
      # display.binary(defGlove, ratio=0.5, window_name="oldGlove")
      #
      #kernel = np.ones((5,5), np.uint8)
      #defGlove = cv2.erode(defGlove.astype('uint8'), kernel, 3)
      #
      # scipy: no good.  requires too much dilation.
      # probably best to go back to watershed and be aggressive.
      #kernel = np.ones((5,5), np.uint8)
      #defGlove = scipy.ndimage.binary_dilation(defGlove.astype('uint8'), kernel, 10)
      #defGlove = np.logical_and(defGlove, gDet.fgIm)
      #display.binary(defGlove, ratio=0.5, window_name="newGlove")
      
    #
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



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

  #----------------------------- emptyState ----------------------------
  #
  def emptyState(self):
    '''!
    @brief      Get and empty state to recover its basic structure.

    @param[out]  estate     The empty state.
    '''

    eState = StateDetectors()
    pass #for now. just getting skeleton code going.

  #------------------------------ getState -----------------------------
  #
  def getState(self):
    '''!
    @brief      Get the complete detector state, which involves the 
                states of the individual layer detectors.

    @param[out]  state  The detector state for each layer, by layer.
    '''

    cState = StateDetectors()

    gDet = self.glove.getState()

    cState.x      = 150*self.imGlove + 75*self.imPuzzle 
    cState.hand   = self.imGlove
    cState.pieces = self.imPuzzle
    cState.isHand = self.hand

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

    The save process saves the necessary information to re-instantiate
    a PuzzleDetectors class object. 

    @param[in] fPtr    An HDF5 file point.
    '''

    # Recursive saving to contained elements. They'll make their
    # own groups.
    self.workspace.saveTo(fPtr)
    self.depth.saveTo(fPtr)
    self.glove.saveTo(fPtr)

    if (self.mask is not None):
      fPtr.create_dataset("theMask", data=self.mask)

  #
  #-----------------------------------------------------------------------
  #============================ Static Methods ===========================
  #-----------------------------------------------------------------------
  #

  #============================ buildFromCfg ===========================
  #
  @staticmethod
  def buildFromCfg(theConfig):
    '''!
    @brief  Instantiate from stored configuration file (YAML).
    '''
    theDet = PuzzleDetectors(theConfig)

  #================================ load ===============================
  #
  @staticmethod
  def load(inFile):
    fptr = h5py.File(inFile,"r")
    theDet = PuzzleDetectors.loadFrom(fptr)
    fptr.close()
    return theDet

  #============================== loadFrom =============================
  #
  def loadFrom(fPtr):
    # Check if there is a mask

    fgGlove = Glove.fgGaussian.loadFrom(fPtr)
    wsColor = inCorner.inCorner.loadFrom(fPtr)
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

    detPS = PuzzleDetectors(None, detFuns, None)
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
    # @todo Need to have these hard-coded values be parameters in the
    #       config dict.  (-105,0), (35), 
    #           mu    = np.array([150.0,2.0,30.0]), 
    #           sigma = np.array([1100.0,250.0,250.0]) )
    #
    bgModel    = inCorner.inCorner.build_model_blackBG(-105, 0)
    bgDetector = inCorner.inCornerEstimator()

    bgDetector.set_model(bgModel)
    bgDetector.refineFromStreamRGBD(theStream, True)

    #==[2]  Step 2 is to get the largest region of interest as a 
    #       workspace mask.  Then apply margins generated from refinement
    #       processing in the earlier step.
    #
    theMask = bgDetector.maskRegionFromStreamRGBD(theStream, True)

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
    print("\nThis step is for the depth model: count to 2 then quit.")
    theConfig = onWorkspace.CfgOnWS.builtForPuzzlebot()
    bgModel   = onWorkspace.onWorkspace.buildAndCalibrateFromConfig(theConfig, \
                                                                    theStream, True)

    #==[4]  Step 3 is to get the foreground color model.
    #
    print("\nThis step is for the glove model.")
    fgModP  = Glove.SGMdebug(mu    = np.array([150.0,2.0,30.0]),
                             sigma = np.array([1100.0,250.0,250.0]) )
    fgModel = Glove.fgGaussian( Glove.CfgSGT.builtForRedGlove(), None, fgModP )

    fgModel.refineFromStreamRGBD(theStream, True)


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
    detFuns = InstPuzzleScene(workspace_color = bgDetector,
                              workspace_depth = bgModel,
                              workspace_mask  = theMask,
                              glove           = fgModel)
    
    detPS = PuzzleDetectors(None, detFuns, None)
    detPS.save(outFile)

    # CODE FROM LAYERED DETECTOR CONSTRUCTOR.  WILL BUILD ON OWN FROM
    # CONFIGURATION.  DOES NOT ACCEPT BUILT INSTANCES. ONLY OPTION IS
    # TO SAVE THEN LOAD UNLESS THIS CHANGES.
    #
    #self.workspace = inCorner.inCorner.buildFromCfg(detCfg.workspace.color)
    #self.depth     = onWorkspace.onWorkspace.buildFromCfg(detCfg.workspace.depth)
    #self.glove     = Glove.fgGaussian.buildFromCfg(detCfg.glove)



#================================= HandByDepth =================================


class HandByDepth(detBase.inImageRGBD):
  '''!
  @ingroup  Surveillance
  @brief    Detector based on depth only, no glove appearance model.

  @note     We still call it a glove to maintain compatibility with the
            glove-based model.  In principle, we should call it HandByDepth.
            Will have to figure that out eventually.  Do we rigidly stick to
            Glove or do we move to Hand and then augment with glove
            capabilities?  Or, do we modify internal member variables to refer
            to hand so that it make sense whether or not there is a glove on
            the hand?  Need to resolve. 2025/07/22 PAV.

  @note     Changing to HandByDepth and committing to eventually aligning
            member variables with this change.  Right now, need to recall
            how everything was implemented and do better job at documenting
            things for easier development in the future. 2027/07/22 PAV.

  @note     How can we generalize this so that a general robot can use it
            and even employ tracking to interpret and learn from human
            (demonstrator)? 2025/07/22 PAV.
  '''

  def __init__(self, detCfg = None, detInst = None, processors=None):
    '''!
    @brief      Constructor for layered puzzle scene detector assuming
                no glove is being used.

    @param[in]  detCfg      Detector configuration.
    @param[in]  processors  Image processors for the different layers.
    @param[in]  detInst     Detection instances for the different layers.
    '''
    
    super(HandByDepth,self).__init__(processors)

    if (detCfg is None):
      detCfg = CfgPuzzleScene()

    if (detInst is not None):

      self.workspace = detInst.workspace_color 
      self.depth     = detInst.workspace_depth

      if (detInst.workspace_mask is not None):
        self.mask   = detInst.workspace_mask

    else:
      self.workspace = inCorner.inCorner.buildFromCfg(detCfg.workspace.color)
      self.depth     = onWorkspace.onWorkspace.buildFromCfg(detCfg.workspace.depth)
      if (detCfg.workspace.mask is not None):
        self.mask   = detCfg.workspace.mask

    self.imHand   = None
    self.imPuzzle = None
    self.hand     = False

    self.params   = detCfg


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

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   An RGB-D image (structure/dataclass).
    '''

    ##  First, perform any specified pre-processing.
    #
    # @note Not dealing with pre-processor, but it might be important.
    # @todo Figure out how to use the improcessor.
    #

    ##  Second, invoke the layer detectors and post-processor to differentiate
    ##  the actual semantic layers of the scene.  The layer detectors should be
    ##  considered as raw detectors that need further polishing to extract the
    ##  desired semantic layer information.  These layers are further processed
    ##  by customized track pointers and filters.
    #
    self.workspace.measure(I.color)
    self.depth.measure(I.depth)

    cDet = self.workspace.getState()    # Binary mask region of puzzle mat.
    dDet = self.depth.getState()        # Binary mask region close to planar surface.
    ##  The post processing here is hard-coded rather than a private member function
    ##  invocation.  
    #
    # The hand regions above the surface get recovered here.  Unfortunately, the
    # Realsense is not the best depth detector and fails to capture regions near
    # the work surface.  The best thing to do is to expand the too high regions,
    # to excise nearby areas from workmat detection region.
    # 
    # Again, "level with the surface" is only accurate up to the depth camera's
    # depth sensitivity.  If not the Realsense, the depth might still be a
    # little too noisy to really capture fine details. 
    #
    tooHigh = np.logical_not(dDet.bgIm)

    if (self.mask is not None):
      np.logical_and(tooHigh, self.mask, out=tooHigh)

    count     = np.count_nonzero(tooHigh)
    # @todo Make the nnz count threshold a parameter. For now hard coded.
    #       Check if already done in the YAML file.
    if (count < 500):
      # Zero out glove regions. It is not present or not in active area (moving out of
      # the field of view, or just entered but not quite fully in).
      tooHigh.fill(False)
    else:
      num_hi, lab_hi, stats_hi, cent_hi = \
        cv2.connectedComponentsWithStats(tooHigh.astype(np.uint8), connectivity=8)
      if num_hi > 1:
        k_big_hi = 1 + np.argmax(stats_hi[1:, cv2.CC_STAT_AREA])
        tooHigh = (lab_hi == k_big_hi)

    kernel  = np.ones((3,3), np.uint8)
    scipy.ndimage.binary_dilation(tooHigh, kernel, 7, output=tooHigh)

    # Recount. Hard coded temporarily
    count     = np.count_nonzero(tooHigh)
    self.hand = (count > 30000)

    ##  Package the processed layers started with too high.  Next, remove any
    ##  parts of the not surface layer that intersect with the expanded glove 
    ##  region.  May remove adjacent puzzle piece area; that's OK since we can't
    ##  rely on those pieces having been fully measured/captured.
    ##  After that
    #
    SurfaceButNotMat = np.logical_and(np.logical_not(cDet.x), np.logical_not(tooHigh))
    if (self.mask is not None):
      np.logical_and(SurfaceButNotMat, self.mask, out=SurfaceButNotMat)

    # Connects touched pieces a little easier.  Helps with preventing break up though.
    scipy.ndimage.binary_closing(SurfaceButNotMat, kernel, 1, output = SurfaceButNotMat)

    self.imHand   = tooHigh
    self.imPuzzle = SurfaceButNotMat

    #DEBUG VISUALIZATION - EVERY LOOP
    #display.binary(dDet.bgIm,window_name="too high")

    #
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



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

  #----------------------------- emptyState ----------------------------
  #
  def emptyState(self):
    '''!
    @brief      Get empty puzzle scene detector state to recover its basic structure.

    Translated from Matlab, which required access to this in advance
    to manage streaming data storage. Python isn't as strongly typed.

    @param[out]  estate     The empty state.
    '''

    eState = StateDetectors()
    return eState

  #------------------------------ getState -----------------------------
  #
  def getState(self):
    '''!
    @brief      Get the complete detector state, which involves the 
                states of the individual layer detectors.

    @param[out]  state  The detector state for each layer, by layer.
    '''

    cState        = StateDetectors()
    cState.x      = 150*self.imHand + 75*self.imPuzzle 
    cState.hand   = self.imHand
    cState.pieces = self.imPuzzle
    cState.isHand = self.hand

    return cState

  #----------------------------- emptyDebug ----------------------------
  #
  def emptyDebug(self):
    '''!
    @brief      Get empty puzzle scene detector debug state.
    '''

    return None     # @note None for now. just getting skeleton code going.

  #------------------------------ getDebug -----------------------------
  #
  def getDebug(self):
    '''!
    @brief      Get debug information if available.

    Status: Not available.
    '''

    return None     # @note None for now. just getting skeleton code going.

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
    a HandByDepth class object. 

    @param[in] fPtr    An HDF5 file point.
    '''

    # Recursive saving to contained elements. They'll make their
    # own groups.
    self.workspace.saveTo(fPtr)
    self.depth.saveTo(fPtr)

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
    theDet = HandByDepth(theConfig)

  #================================ load ===============================
  #
  @staticmethod
  def load(inFile):
    fptr = h5py.File(inFile,"r")
    theDet = HandByDepth.loadFrom(fptr)
    fptr.close()
    return theDet

  #============================== loadFrom =============================
  #
  def loadFrom(fPtr):
    # Check if there is a mask

    wsColor = inCorner.inCorner.loadFrom(fPtr)
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
                              glove           = None)

    detPS = HandByDepth(None, detFuns, None)
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
  # is not fully instantiated.  
  #
  # @param[in] theStream    Aligned RGBD stream.
  # @param[in] outFile      Full path filename of HDF5 configuration output.
  #
  @staticmethod
  def calibrate2config(theStream, outFile):

    #==[1]  Step 1 is to get the background color model.
    #       Hardcoded initial configuration with some refinement.
    #
    # @todo Need to have these hard-coded values be parameters in the
    #       config dict.  (-105,0), (35), 
    #           mu    = np.array([150.0,2.0,30.0]), 
    #           sigma = np.array([1100.0,250.0,250.0]) )
    #
    bgModel    = inCorner.inCorner.build_model_blackBG(-105, 0)
    bgDetector = inCorner.inCornerEstimator()

    bgDetector.set_model(bgModel)
    bgDetector.refineFromStreamRGBD(theStream, True)

    #==[2]  Step 2 is to get the largest region of interest as a 
    #       workspace mask.  Then apply margins generated from refinement
    #       processing in the earlier step.
    #
    theMask = bgDetector.maskRegionFromStreamRGBD(theStream, True)

    kernel  = np.ones((3,3), np.uint8)
    scipy.ndimage.binary_erosion(theMask, kernel, 2, output=theMask)

    bgDetector.apply_estimated_margins()
    bgDetector.bgModel.offsetThreshold(35)

    #==[3]  Step 4 is to get the depth workspace model.
    #
    print("\nThis step is for the depth model: count to 2 then quit.")
    theConfig = onWorkspace.CfgOnWS.builtForPuzzlebot()
    bgModel   = onWorkspace.onWorkspace.buildAndCalibrateFromConfig(theConfig, \
                                                                    theStream, True)

    #==[4]  Step 4 is to package up and save as a configuration.
    #       It involves instantiating a layered detector then
    #       saving the configuration.
    #   OR
    #       Manually saving as HDF5, possibly with YAML config string.
    #       Anything missing will need to be coded up.
    #
    detFuns = InstPuzzleScene(workspace_color = bgDetector,
                              workspace_depth = bgModel,
                              workspace_mask  = theMask,
                              glove           = None)
    
    detPS = HandByDepth(None, detFuns, None)
    detPS.save(outFile)

#
#-------------------------------------------------------------------------
#============================= Trackpointers =============================
#-------------------------------------------------------------------------
#

@dataclass
class StatePuzzleTracks:
  '''!
  @ingroup  Surveillance
  @brief    Basic puzzle trackpointer state information.

  The state information consists of:
  field  | contents
  ------ | --------
  handPt    | Coordinate location of hand/glove.
  pcsPts    | Coordinate location of disjoint puzzle pieces regions.
  isHand    | Boolean indicating that hand is in scene.
  arePieces | Boolean indicating existence of pieces in scene.
  '''
  handPt    : any = None        
  pcsPts    : any = None        
  pcsIds    : any = None
  isHand    : bool = False      
  arePieces : bool = False

class TrackPointers(object):
  '''!
  @ingroup  Surveillance
  @brief    Track pointers for the glove/hand and the puzzle pieces.
  '''

  def __init__(self, iState = None, trackCfg = None):
    '''!
    @brief  Constructor for layered puzzle scene tracker.

    @param[in]  iState      Initial state of tracks.
    @param[in]  trackCfg    Trackpointer(s) configuration.
    '''
    
    # @note Regarding the PuzzleScene Trackpointers, 
    # Will most likely need to differentiate in play vs placed pieces.
    # May be part of the puzzle filter and not here at the track pointer
    # level.  Also may be pushed to a separate ROS thread.  Publisher
    # will send along the necessary data for processing.
    # @note Yes, push to filter / association since that knows what's up.
    #
    #self.piecesInPlay = trackpointer.centroidMulti
    #self.piecesPlaced = trackpointer.centroidMulti
    if (trackCfg is None):
      puzzleCfg = CfgPuzzleScene()
      trackCfg  = puzzleCfg.pieces
      
    self.pieces = tpieces.centroidMulti(params=trackCfg)
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

    self.pieces.predict()
    self.glove.predict()

  #------------------------------ measure ------------------------------
  #
  def measure(self, I):
    '''!
    @brief  Apply detection to the source image pass.

    @param[in]  I   Layered detection image instance (structure/dataclass).
    '''

    self.glove.measure(I.hand)
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

  #----------------------------- emptyState ----------------------------
  #
  def emptyState(self):
    '''!
    @brief       Get empty puzzle scene track state.

    @param[out]  estate     The empty state.
    '''

    estate = StatePuzzleTracks()
    return estate

  #------------------------------ getState -----------------------------
  #
  def getState(self):
    '''!
    @brief      Get the complete detector state, which involves the 
                states of the individual layer detectors.

    @param[out]  state  The detector state for each layer, by layer.
    '''

    cstate = StatePuzzleTracks()

    tstate = self.glove.getState()
    pstate = self.pieces.getState()

    cstate.handPt    = tstate.tpt
    cstate.isHand    = tstate.haveMeas
    cstate.pcsPts    = pstate.tpt
    cstate.arePieces = pstate.haveMeas
    cstate.pcsIds    = None

    return cstate

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
  def display_cv(self, I, ratio = None, window_name="trackpoints", doRotate = False):
    
    if ratio is None:
      msize = 10
      mthick = 2
    elif ratio < 1:
      msize = (np.fix(10/ratio)).astype(int)
      mthick = (np.fix(2/ratio)).astype(int)
    else:
      msize = 10
      mthick = 2

    if (self.glove.haveMeas):

      Imark = display.annotate_trackpoint(I, self.glove.tpt, (255,255,255), msize, mthick)
      if (self.pieces.haveMeas):
        Imark = display.annotate_trackpoints(Imark, self.pieces.tpt, (255,0,0), msize, mthick)

    else:

      if (self.pieces.haveMeas):
        Imark = display.annotate_trackpoints(I, self.pieces.tpt, (255,0,0), msize, mthick)
      else:
        Imark = I

    if doRotate:
      Imark = cv2.rotate(Imark, cv2.ROTATE_180)

    display.rgb(Imark, ratio, window_name)



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
#============================ PuzzlePerceiver ============================
#-------------------------------------------------------------------------
#
@dataclass
class InstPuzzlePerceiver():
    '''!
    @brief Class for collecting visual processing methods needed by the
    PuzzleScene perceiver.

    '''
    detector : any
    trackptr : any
    trackfilter : any
    #to_update : any    # What role/purpose??

@dataclass
class StatePuzzleScene():
  '''!
  @ingroup  Surveillance
  @brief    Perceiver puzzle scene state aggegating detector and tracker,
            plus filter as fitting, information.

  Contents of this dataclass are:
  field    | description
  -------- | -----------
  segIm    | Segmentation image of hand + puzzle pieces regions.
  hand     | Hand track point.
  puzzle   | Puzzle pieces track points.
  isHand   | Is hand in scene?
  isPuzzle | Are there puzzle pieces in the scene?
  '''
  segIm     : any
  hand      : any
  puzzle    : any
  pieceIds  : any
  isHand    : bool = False
  isPuzzle  : bool = False

class PuzzlePerceiver(Perceiver):
  '''!
  @ingroup  Surveillance
  @brief    Perceiver based on glove and work scene/mat detection models.

  Usually the detectors will ignore the robot arm and even parts of the human
  arm.  They act as nusiance elements to the perceiver and subsequent
  processing, which should be cognizant of that fact.
  '''

  #============================== __init__ =============================
  #
  def __init__(self, perCfg = None, perInst = None):
    '''!
    @brief  Constructor for a PuzzlePerceiver.

    @param[in]  perCfg      Perceiver configuration.
    @param[in]  perInst     Perceiver component instances, if already created.
    '''

    if perInst is not None:
      super().__init__(perCfg, perInst.detector, perInst.trackptr, perInst.trackfilter)
    else:
      raise Exception("Sorry, not yet coded up.") 
      # @todo   Presumably contains code to instantiate detector, trackptr, filter, etc.
    

  #============================== predict ==============================
  #
  def predict(self):
    self.detector.predict()
    if (self.filter is not None):
      self.filter.predict()

  #============================== measure ==============================
  #
  def measure(self, I):
    # First perform detection.
    self.detector.measure(I)

    # Get state of detector. Pass on to trackpointer.
    dState = self.detector.getState()
    self.tracker.process(dState)

    # MAKING THIS UP.  GOTTEN FROM PerceiveGloveBC
    tGlove = self.tracker.glove.getState()
    self.haveRun   = True
    self.haveState = True   # WHAT IS THIS?
    if (tGlove is not None):
      #print(tGlove)
      self.haveObs   = tGlove.haveMeas
      if tGlove.haveMeas:
        self.tMeas = tGlove.tpt;
    else:
      self.haveObs = False
      self.tMeas   = None

    #IAMHERE.  THIS CLASS DOES NOT PROPERLY CONSTRUCT A STATE.
    #LOOK AT THE GLOVEBYCOLOR PERCEIVER TO KNOW HOW TO DO SO
    #IN A CLEAN WAY, THEN TO PASS IT ON TO THE MONITOR IN A 
    #REASONABLE MANNER.  THAT'S NOT DONE HERE.
    #
    #WE ALSO NEED TO CREATE A CUSTOM ACTIVTY REGION PROCESSOR
    #IF IT EXPECTS TO GET GIVEN A DIRECT OUTPUT.  OR WE NEED TO
    #DEFINE A SPECIAL PREPROCESSOR THAT PULLS OUT WHAT IS NEEDED.
    #THIS LATTER SEEMS LIKE A BETTER IDEA.
    # 
    #2025/07/22 PAV.
    #
    # If there is a filter, get track state and pass on to filter.

  #============================== correct ==============================
  #
  def correct(self):
    if (self.filter is not None):
      trackOut = self.tracker.getOutput()
      self.filter.correct(trackOut)


  #=============================== adapt ===============================
  #
  def adapt(self):
    # @note Not implemented. Deferring to when needed. For now, kicking to filter.
    # @note Should have config flag that engages or disengages, or member variable flag.
    if (self.filter is not None):
      self.filter.adapt()

    pass

  #============================== process ==============================
  #
  def process(self, I):
    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

    pass

  #=============================== detect ==============================
  #
  # IS this really needed??? Isn't it already done in measure?
  def detect(self):
    pass

  #============================= emptyState ============================
  #
  def emptyState(self):
    '''!
    @brief  Return empty puzzle scene state information.
    '''
    pass

  #============================== getState =============================
  #
  def getState(self):
    '''!
    @brief  Get puzzle scene state information.

    For the puzzle state, we've got the detector information, the
    hand/glove track state, and the puzzle piece track states.
    Let's package them all up as best as possible.
    '''

    dstate = self.detector.getState()
    tstate = self.tracker.getState()

    if (self.filter is None):
      cState = StatePuzzleScene( segIm  = dstate.x, 
                                 hand   = tstate.handPt,
                                 puzzle = tstate.pcsPts,
                                 isHand = tstate.isHand,
                                 isPuzzle = tstate.arePieces,
                                 pieceIds = None) 
    else: 
      # @todo   Placeholder for filtered state.  Same as tracked state.
      #         Filered would return pieceIds list as association over
      #         time is managed.
      cState = StatePuzzleScene( segIm = dstate.x, 
                                 hand  = tstate.handPt,
                                 puzzle = tstate.pcsPts,
                                 isHand = tstate.isHand,
                                 isPuzzle = tstate.arePieces,
                                 pieceIds = None) 

    return cState


  #============================= emptyDebug ============================
  #
  def emptyDebug(self):
    pass

  #=========================== getDebugState ===========================
  #
  def getDebugState(self):
    pass

  #============================= display_cv ============================
  #
  # @brief  Display any found track points on passed (color) image.
  #
  #
  def display_cv(self, I, ratio = None, window_name="puzzle pieces"):
    
    if (self.filter is None):

      if false and (self.tracker.haveMeas):
        display.trackpoints_cv(I, self.tracker.tpt, ratio, window_name)
      else:
        display.rgb_cv(I, ratio, window_name)

    else:

      not_done()

#  #======================= buildWithBasicTracker =======================
#  #
#  # @todo   Should this be packaged up more fully with tracker config?
#  #         Sticking to only detConfig is not cool since it neglects
#  #         the tracker.
#  #
#  @staticmethod
#  def buildWithBasicTracker(buildConfig):
#    """!
#    @brief  Given a stored detector configuration, build out a puzzle
#            perceiver with multi-centroid tracking.
#
#    Most of the configuration can default to standard settings or to
#    hard coded puzzle settings (that should never be changed).
#    """
#
#    print(buildConfig)
#    if (buildConfig.tracker is None):
#      buildConfig.tracker = defaults.CfgCentMulti()
#
#    if (buildConfig.perceiver is None):
#      buildConfig.perceiver = perBase.CfgPerceiver()
#
#    if (isinstance(buildConfig.detector, str)):
#      matDetect    = Detector.load(buildConfig.detector)
#    elif (isinstance(buildConfig.detector, CfgDetector)):
#      matDetect    = Detector(buildConfig.detector)
#    elif (buildConfig.detector is None):
#      matDetect    = Detector()
#    else:
#      warnings.warn('Unrecognized black work mat detector configuration. Setting to default.')
#      matDetect    = Detector()
#
#    piecesTrack  = tracker.centroidMulti(None, buildConfig.tracker)
#    piecesFilter = None
#
#    perInst     = InstPuzzlePerceiver(detector = matDetect, 
#                                      trackptr = piecesTrack,
#                                      trackfilter = piecesFilter)
#
#    return PuzzlePerceiver(buildConfig.perceiver, perInst)



#
#-------------------------------------------------------------------------
#============================ PuzzleCalibrator ===========================
#-------------------------------------------------------------------------
#

class PuzzleCalibrator(PuzzleDetectors):
  '''!
  @ingroup  Surveillance
  @brief    Detection calibrator: Possible that not used due to static
            calibration methods in the Detector classes proper.
  '''

  # @todo Need to flip: config, instances, processors. Align with super class.
  def __init__(self, detCfg = None, processors=None, detModel = None):
    '''!
    @brief  Constructor for layered puzzle scene detector.

    @param[in]  detCfg      Detector configuration.
    @param[in]  processors  Image processors for the different layers.
    @param[in]  detModel    Detection models for the different layers.
    '''
    
    super(PuzzleCalibrator,self).__init__(processors)

    self.workspace = detector.bgmodel.inCornerEstimator()
    self.depth     = detector.bgmodel.onWorkspace()
    self.glove     = detector.fgmodel.fgGaussian()

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
    #tinfo.name = mfilename; #tinfo.version = '0.1;';
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
#-------------------------------------------------------------------------------
#=============================== PuzzleActivities ==============================
#-------------------------------------------------------------------------------
#

class ZoneCounter(Counter):
  def __add__(self, other):
    if not isinstance(other, Counter):
      return NotImplemented
    result = ZoneCounter()
    for elem, count in self.items():
      newcount = count + other[elem]
      result[elem] = newcount
    for elem, count in other.items():
      if elem not in self:
        result[elem] = count
    return result


@dataclass
class StatePuzzleActivity():
  '''!
  @ingroup  Surveillance
  @brief    Perceiver puzzle scene activity state structure. 

  Contents of this dataclass are:
  field    | description
  -------- | -----------
  hand     | Activity state of hand.
  puzzle   | State of puzzle. 
  '''
  hand     : any
  puzzle   : any
  zones    : ZoneCounter


  #=========================== getZoneCounts ===========================
  #
  #
  def getZoneCounts(self):
    zsort = sorted(self.zones.keys())
    return [self.zones[z] for z in zsort]


class PuzzleActivities(imageRegions):
  '''!
  @ingroup  Surveillance
  @brief    Simple puzzle activity monitor based on specified image regions.

  Purpose of this class is to show how to integrate an activity detector into
  the puzzle scene monitor in a way compatible with the puzzle scene perceiver.
  For more specialized processing, create a sub-class.
  '''

  #=========================== PuzzleActivites ===========================
  #
  #
  def __init__(self, imRegions): 
    """!
    @brief  Constructor for simple puzzle scene activity detector.
  
    @param[in]    imRegions   Label-type image.
    """

    super(PuzzleActivities,self).__init__(imRegions)

  #============================== measure ==============================
  #
  def measure(self, y):
    """
    @brief  Compare signal to expected image region states.

    @param[in]  zsig  The 2D pixel coords / 3D pixel coords + depth value.
    """
    noCounts = ZoneCounter({x:0 for x in range(self.lMax+1)})
    self.z = StatePuzzleActivity(hand = [-1], puzzle = None, zones = noCounts)

    if not self.isInit:
      return

    if y.isHand: 
      # Map coordinates takes in (i,j). Map zsig from (x,y) to (i,j).
      yhand  = np.flipud(y.hand)
      self.z.hand = scipy.ndimage.map_coordinates(self.imRegions, yhand, order = 0)
    else:
      self.z.hand = [-1]

    if y.isPuzzle:
      ypuzz  = np.flipud(y.puzzle)
      self.z.puzzle  = scipy.ndimage.map_coordinates(self.imRegions, ypuzz, order = 0)
      self.z.zones = ZoneCounter(self.z.puzzle) + noCounts
    else:
      pass

  #=============================== process ===============================
  #
  def process(self, x):
    """!
    @brief  Run entire processing pipeline.

    The entire pipeline consists of predict, measure, correct, and adapt. At least
    if there is a measurement.  If no measurement, then only predict is executed
    since there is no measurement to interpret, correct, and adapt with.
    """
    self.predict()
    if x.isHand or x.isPuzzle:
      self.measure(x)
      self.correct()
      self.adapt()


  #================================ load ===============================
  #
  @staticmethod
  def load(fileName, relpath = None):    # Load given file.
    """!
    @brief  Outer method for loading file given as a string (with path).

    Opens file, preps for loading, invokes loadFrom routine, then closes.
    Overloaded to invoke coorect loadFrom member function.

    @param[in]  fileName    The full or relative path filename.
    @param[in]  relpath     The hdf5 (relative) path name to use for loading.
                            Usually class has default, this is to override.
    """
    print(fileName)
    fptr = h5py.File(fileName,"r")
    if relpath is not None:
      theInstance = PuzzleActivities.loadFrom(fptr, relpath);
    else:
      theInstance = PuzzleActivities.loadFrom(fptr)

    fptr.close()
    return theInstance

  #============================== loadFrom =============================
  #
  @staticmethod
  def loadFrom(fptr, relpath="activity.byRegion"):
    """!
    @brief  Inner method for loading internal information from HDF5 file.

    Load data from given HDF5 pointer. Assumes in root from current file
    pointer location.
    """
    gptr = fptr.get(relpath)

    keyList = list(gptr.keys())
    if ("imRegions" in keyList):
      regionsPtr = gptr.get("imRegions")
      imRegions  = np.array(regionsPtr)
    else:
      imRegions  = None

    theDetector = PuzzleActivities(imRegions)

    return theDetector


#
#-------------------------------------------------------------------------------
#================================ PuzzleMonitor ================================
#-------------------------------------------------------------------------------
#

class PuzzleMonitor(Monitor):
  '''!
  @ingroup  Surveillance
  @brief    Puzzle monitor that examines hand/glove state and puzzle state, or
            equivalent information as measured by a perceiver.

  A generic sub-class of Monitor intended to demonstrate how to implement with
  a richer output signal due to the layers.  It might serve well for a diverse
  set of implementations as the role of the monitor may be to shepherd information
  around.  Some cases of specialized processing may require creating a sub-class.
  '''

  #============================ PuzzleMonitor ============================
  #
  #
  def __init__(self, theParams, thePerceiver, theActivity, theReporter = None):
    """!
    @brief  Constructor for the perceiver.monitor class.
  
    @param[in] theParams    Option set of paramters. 
    @param[in] thePerceiver Perceiver instance (or possibly not).
    @param[in] theActivity  Activity detector/recognizer.
    @param[in] theReporter  Reporting mechanism for activity outputs.
    """

    super(PuzzleMonitor,self).__init__(theParams, thePerceiver, theActivity, theReporter)

    # See documentation for the Monitor to gauge code stability.

  #=============================== process ===============================
  #
  #
  def process(self, I):
    """!
    @brief  Run perceive + activity recognize pipeline for one step/image
            measurement.
    """

    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

    self.reporter.process(self.getState())

#  #============================ displayState ===========================
#  #
#  def displayState(self, dState = None):
#    """!
#    @brief  Display the perceiver state and activity state per configuration
#            specification.
#
#    @param[in]  dState  Monitor state to display (optional). Default is current state.
#    """
#
#    if (self.params.display == 'basic'):
#      if dState is None: 
#        self.perceiver.displayState()
#        self.activity.printState()
#      else:
#        self.perceiver.displayState(dState.perceiver)
#        self.activity.printState(dState.activity)
#
#    elif (self.params.display == 'overlay'):
#      # @todo Need to implement.  Requires window name.  Not an argument.
#      #       For now do not invoke this version.
#      if dState is None: 
#        self.perceiver.displayState()
#        self.activity.displayState()
#      else:
#        self.perceiver.displayState(dState.perceiver)
#        self.activity.displayState(dState.activity)
#
#  #============================ displayDebug ===========================
#  #
#  def displayDebug(self, dbState = None):
#    """!
#    @brief  Display the debug state. Punts to contained instances.
#    """
#    if (params.displayDebug == 'basic'):
#      if dState is None: 
#        self.perceiver.displayState()
#        self.activity.printState()
#      else:
#        self.perceiver.displayState(dState.perceiver)
#        self.activity.printState(dState.activity)
#
#    elif (params.display == 'overlay'):
#      # @todo Need to implement.  Requires window name.  Not an argument.
#      #       For now do not invoke this version.
#      if dState is None: 
#        self.perceiver.displayState()
#        self.activity.displayState()
#      else:
#        self.perceiver.displayState(dState.perceiver)
#        self.activity.displayState(dState.activity)

  #================================ info ===============================
  #
  #
  def info(self):
    """!
    @brief      Return the information structure used for saving or
                otherwise determining the tracker setup for
                reproducibility.
   
    @param[out] tinfo   The tracking configuration information structure.
    """

    return None
    #tinfo = Info(name=os.path.basename(__file__),
    #     version='1.0.0',
    #     data=time.strftime('%Y/%m/%d'),
    #     time=time.strftime('%H:%M:%S'),
    #     params=self.params)

    #return tinfo

  #================================= free ================================
  #
  #
  def free(self):
    """!
    @brief      Destructor.  Just in case other stuff needs to be done.
    """
    pass

  # @todo Eventually make these member functions protected and not public.

  #=============================== measure ===============================
  #
  #
  def measure(self, I):
    """!
    @brief  Run activity detection process to generate activity state measurement. 
            If perceiver has no measurement/observation, then does nothing.

    @param[in]  I   Image to process. Depending on implementation, might be optional.
    """

    if not self.params.external:    # Perceiver process not externally called.
      self.perceiver.process(I)     # so should run perceiver process now.

    pstate = self.perceiver.getState()

    self.activity.process(pstate)

    # do post processing to collect what is needed.

#
#============================== PuzzleScene ==============================
