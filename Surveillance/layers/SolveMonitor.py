#============================== SolveMonitor ==============================
##
# @package  Surveillance.layers.SolveMonitor
#
# @brief    Monitor systems with activity classes for monitoring human 
#           solve experiment i.e taking pieces from outside solution 
#           region and placing them in.
#
# @ingroup  Surveillance
#
# @author   Nihit Agarwal,   nagarwal90@gatech.edu
# @date     2026/04/20
#
#============================== SolveMonitor ==============================

#--[0.A] Standard python libraries.
#
import numpy as np
import scipy
import rospy
from dataclasses import dataclass
import h5py
from skimage.segmentation import watershed
from skimage import morphology
from skimage import measure
from Surveillance.layers.PuzzleScene import *
import matplotlib.pyplot as plt
import time
from detector.base import DetectorState
from typing import ClassVar, List
from puzzle.builder.arrangement import Arrangement, CfgArrangement
import puzzle.board as board
from Surveillance.layers.PuzzleMonitors import Piece




@dataclass
class SolveState:
  '''!
  @ingroup  Surveillane
  @brief    Capture segmented scene binary mask
            for each zone along with occlusion state.
  field | description
  ______| ______________
  regions | binary mask of each zone
  zoneOcc | occlusion state of each zone
  '''
  regions: any
  zoneOcc: List[int]
  totalPieces: int
  piecesPlaced : int=0


@dataclass
class SolveOutput:
  '''!
  @ingroup  Surveillance
  @brief    Cosolve Activity Monitor output 

  Contents of this dataclass are:
  field    | description
  -------- | -----------
  rgb        | RGB capture after placement of piece
  pcInfo     | Details of piece placed
  actor      | Agent performing action
  haveObs    | flag to set when observation obtained
  '''
  rgb : any
  pcInfo : any
  actor : str
  haveObs : bool

@dataclass
class SolveInput:
  '''!
  @ingroup  Surveillance
  @brief    Data class to capture the puzzle scene detector
            state and raw input signal to be fed to sort
            activity monitor

  Contents of this dataclass are:
  field      | description
  --------   | -----------
  sceneState | Detector state of scene perceiver 
  image      | Raw image signal.
  '''
  sceneState: StatePuzzleScene
  image: any




#
#-----------------------------------------------------------------------------
#================================ SolveActivity ==============================
#-----------------------------------------------------------------------------
#

class SolveActivity(PuzzleActivities):
  """!
  @brief  Class detects the state of the cosolve process
  """

  #============================== __init__ =============================
  #
  def  __init__(self, imRegions):
    super().__init__(imRegions)

    # Set calibrate param to false
    self.calibrated = False
    self.piece_threshold = 100 # min number of pixels to consider piece 
                               # appearance or disappearance
    self.hand_threshold = 1 # min number of pixels to consider hand presence
    # Set internal state
    self.x = SolveState(regions=[None] * (self.lMax + 1), zoneOcc=[0] * (self.lMax + 1), piecesPlaced=0, totalPieces=0)
    
    # Set output state
    self.z = SolveOutput(rgb=None, pcInfo=None, actor='', haveObs=False)

    # Set unorganized zone label
    self.unorg = 6

    # Set the solution zone label
    self.soln = 5

    # Set the sol mask
    self.solMask = (self.imRegions == self.soln).astype(int)

    # Set the zones mask
    self.zonesMask = (self.imRegions != 0).astype(int)


    # Puzzle Params
    theParams = CfgArrangement()
    theParams.update(dict(minArea=150))
    theParams.update(dict(maxArea=600))

    # Configuration settings for correspondences
    CfgTrack = board.CfgCorrespondences()
    CfgTrack.matcher = 'SIFTCV'
    CfgTrack.matchParams = None
    CfgTrack.forceMatches = True

    self.cfgTrack = CfgTrack
    self.theParams = theParams

    # Tracker for matching pieces across frames
    self.tracker = None

 

  #============================== measure ==============================
  #
  def measure(self, y: SolveInput):
    """
    @brief  Detect changes in zone occlusion and determine if piece placed.
    @param[in]  y  SolveInput
    """
    if not self.isInit:
      return
    # Reset so that reporter ignores it
    self.z.haveObs = False

    # Initialize var for capturing new state
    zoneOcc = [0] * (self.lMax + 1)
    regions = [None] * (self.lMax + 1)
    update = [False] * (self.lMax + 1)


   # First run calibration
    if not self.calibrated:
        self.calibrated = True
        totalPieces = 0
        for i in range(1, self.lMax + 1):
            mask = (self.imRegions == i).astype(int)
            regions[i] = y.sceneState.segIm * mask
            self.x.regions[i] = regions[i]
            # Count pieces in each zone using regionprops
            labeled = measure.label((regions[i] == 75).astype(int), connectivity=2)
            props = measure.regionprops(labeled)
            # Filter by minimum area threshold
            if i != self.soln:
                piece_count = sum(1 for prop in props if prop.area >= self.piece_threshold)
                totalPieces += piece_count
        self.x.totalPieces = totalPieces
        self.x.piecesPlaced = 0
        return
    
    # Update zone occupancy information
    for i in range(1, self.lMax + 1):
      # check if glove in any zone
      mask = (self.imRegions == i).astype(int)
      regions[i] = y.sceneState.segIm * mask
      # check for 150 in the segmented image which indicates presence of hand
      if np.count_nonzero(regions[i] == 150) > self.hand_threshold:
        zoneOcc[i] = 1
    
    # Compare the segmented image history for each zone
    # if a zone occupancy changes from 1 to 0.

    # Searching for picks and places in all zones
    for i in range(1, self.lMax + 1):
      if self.x.zoneOcc[i] == 1 and zoneOcc[i] == 0:
        # check if piece was removed
        diff = self.x.regions[i] - regions[i]
        clean_diff = np.zeros_like(diff)
        mask = (diff == 75)
        clean_mask = morphology.remove_small_objects(mask, min_size=self.piece_threshold)
        clean_diff[clean_mask] = 75

        # Find the pieces missing, and create regions for them.

        # Region label to get individual pieces
        pick_labeled = measure.label(clean_mask.astype(int), connectivity=2)
        pick_props = measure.regionprops(pick_labeled)

        if len(pick_props) > 1:
          print("ERROS: Detected more than 1 piece pick")
        
        if len(pick_props) > 0:
            prop = pick_props[0]
            pick_y, pick_x = prop.centroid
            print(f"Detected human pick in zone {i} at ({pick_x:.1f}, {pick_y:.1f})")
            self.z.pcInfo = Piece(pick=np.array([pick_x, pick_y]), place=None, zone=i, pick_time=time.time(), place_time=None)
            self.z.actor = 'human'
            update[i] = True
        
        # check if piece was placed
        clean_diff = np.zeros_like(diff)
        mask = (diff == -75)
        clean_mask = morphology.remove_small_objects(mask, min_size=self.piece_threshold)
        clean_diff[clean_mask] = -75
        
        # Region label to get individual placed pieces
        place_labeled = measure.label(clean_mask.astype(int), connectivity=2)
        place_props = measure.regionprops(place_labeled)
        
        if len(place_props) > 1:
            print("ERROR: Detected more than 1 place")
        elif len(place_props) > 0:
            prop = place_props[0]
            place_y, place_x = prop.centroid  
            print(f"Detected human place in zone {i} at ({place_x:.1f}, {place_y:.1f})")
            
            # Only report human places
            if self.z.pcInfo is None:
                print("Warning: Detected place without prior pick")
            else:
                self.z.pcInfo.place = np.array([place_x, place_y])
                self.z.pcInfo.place_time = time.time()
                self.z.haveObs = True
                self.z.rgb = y.image.color
                self.z.actor = 'human'
                self.x.piecesPlaced += 1
            update[i] = True
    # Update internal state
    self.x.zoneOcc = zoneOcc
    for i in range(1, self.lMax + 1):
      if update[i]:
        self.x.regions[i] = regions[i]
    

  
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
    if x.sceneState.isHand or x.sceneState.isPuzzle:
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
      theInstance = SolveActivity.loadFrom(fptr, relpath);
    else:
      theInstance = SolveActivity.loadFrom(fptr)

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

    theDetector = SolveActivity(imRegions)

    return theDetector


#
#-------------------------------------------------------------------------------
#================================ CosolveMonitor ================================
#-------------------------------------------------------------------------------
#

class SolveMonitor(PuzzleMonitor):
  '''!
  @ingroup  Surveillance
  @brief    Sort monitor that examines the workscene perceiver state and
            helps generate sort reports.

  '''

  def measure(self, I):
    """!
    @brief  Run activity detection process to generate activity state measurement. 
            If perceiver has no measurement/observation, then does nothing.

    @param[in]  I   Image to process. Depending on implementation, might be optional.
    """
    if not self.params.external:    # Perceiver process not externally called.
      self.perceiver.process(I)     # so should run perceiver process now.

    pstate = self.perceiver.getState()
    actInput = SolveInput(sceneState=pstate, image=I)
    self.activity.process(actInput)
  
  #=============================== getInternalState ===============================
  #
  #
  def getInternalState(self):
    """!
    @brief  Get internal state of the monitor
    """

    return DetectorState(x=self.activity.x)
