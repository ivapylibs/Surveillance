#============================== CosolveMonitor.py ==============================
##
# @package  Surveillance.layers.CosolveMonitor.py
#
# @brief    Monitor systems with activit detection capabilites which enable
#           tracking of human effort and robot effort in the puzzle solving
#           process.
#
# @ingroup  Surveillance
#
# @author   Nihit Agarwal,   nagarwal90@gatech.edu
# @date     2026/04/21
#
#============================== CosolveMonitor.py ==============================

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
from scipy.signal import convolve2d
from puzzle.piece import PieceStatus


'''
Ideas:

1. Make a puzzle estimate of entire board. 
2. Make a note of whether a piece is visible, occluded, or missing.
3. In each iteration, some pieces will go occluded, some will re-appear. 
4. Match each new piece to the one with closest prior distance.
5. There should be at most 2 pieces which have no match with prior locations.
6. There would be at most 2 piecs which went missing earlier. 
7. Match the piece with no prior with piece which went missing earlier. 
8. For removing attribution confusion, track the last agent that occluded before
   setting as missing. 

'''

@dataclass
class trackPiece:
    VISIBLE = 0
    ROBOT = 1
    HAND = 2
    MISSING = 3
    location: np.ndarray
    state: int
    lastActor: int = -1

@dataclass
class CoSolveState:
  '''!
  @ingroup  Surveillance
  @brief    Capture internal state of the cosolve process. Stores
            the puzzle related data
  field | description
  centroids | List of centroids
  missingPieces | List of missing pieces
  occludedPieces | List of occluded pieces
  
  

  '''
  pieces : List # centroid, state=ROBOT, HAND, VISIBLE
  visible: int
  missing: int
  


@dataclass
class CoSolveOutput:
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
class CoSolveInput:
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
#================================ CosolveActivity ==============================
#-----------------------------------------------------------------------------
#

class CoSolveActivity(PuzzleActivities):
  """!
  @brief  Class detects the state of the cosolve process
  """

  #============================== __init__ =============================
  #
  def  __init__(self, imRegions):
    super().__init__(imRegions)

    # Piece location comparison threshold
    self.distance_threshold = 10

    
    # Set calibrate param to false
    self.calibrated = False
   
    # Set internal state
    self.x = CoSolveState(pieces=[], visible=0, missing=0)
    
    # Set output state
    self.z = CoSolveOutput(rgb=None, pcInfo=None, actor='', haveObs=False)

    # Set unorganized zone label
    self.unorg = 6

    # Set the solution zone label
    self.soln = 5

    # Set the sol mask
    self.solMask = (self.imRegions == self.soln).astype(int)

    # Set the zones mask
    self.zonesMask = (self.imRegions != 0).astype(int)

    # Set the board estimate
    self.board_estimate = None
    
    # Puzzle Params
    theParams = CfgArrangement()
    theParams.update(dict(minArea=150))
    theParams.update(dict(maxArea=600))

    
    self.theParams = theParams
  
  #========== set_solution_board============
  def setSolutionBoard(self, board):
    self.board_estimate = board
    for key in self.board_estimate.pieces:
      self.board_estimate.pieces[key].status = PieceStatus.GONE

  #======================== updateBoardState ===========================
  #
  def updateBoardState(self, segIm):
    solMask = (self.solMask * segIm) == 75
    mask = solMask.astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    convolved = convolve2d(mask, kernel, mode='same')
    threshold = 0.5
    
    added_centers = []
    removed_centers = []
    for key in self.board_estimate.pieces:
      piece = self.board_estimate.pieces[key]
      # Check the average score across the pixel locations
      piece_locations = np.argwhere(piece.y.mask)
      offset = np.array([piece.y.pcorner[1], piece.y.pcorner[0]])
      piece_locations = piece_locations + offset
      
      rows = piece_locations[: ,0]
      cols = piece_locations[:, 1]
      
      score = np.mean(convolved[rows, cols])
      
      

      if score > threshold:
        if self.board_estimate.pieces[key].status != PieceStatus.MEASURED:
          added_centers.append(self.board_estimate.pieces[key].centroidLoc)
          self.board_estimate.pieces[key].setStatus(PieceStatus.MEASURED)
    
    return added_centers     



  #============================== measure ==============================
  #

  def measure(self, y: CoSolveInput):
    """
    @brief  Detect human and robot activity based on scene input
    @param[in]  y  StateCosolveInput
    """
    if not self.isInit:
      return
    # Reset so that reporter ignores it
    self.z.haveObs = False

    rgb = y.image.color
    regions = (self.imRegions > 0) & (self.imRegions != self.soln)
    mask = ((y.sceneState.segIm * regions) == 75)
    puzzleBoard = Arrangement.buildFrom_ImageAndMask(rgb, mask, theParams=self.theParams)

   # First run calibration
    if not self.calibrated:
      self.calibrated = True
      for key in puzzleBoard.pieces:
        piece = puzzleBoard.pieces[key]
        pc = trackPiece(location=piece.centroidLoc, state=trackPiece.VISIBLE)
        self.x.pieces.append(pc)
      self.x.visible = len(self.x.pieces)
      
      # Set the pieces in estimate
      self.updateBoardState(y.sceneState.segIm)
      
      
      return
    
  
    # Extract pieces
    centroids = []
    new_places = []

    old_centroids = [p.location for p in self.x.pieces]
    old_centroids = np.array(old_centroids)
    for key in puzzleBoard.pieces:
        piece = puzzleBoard.pieces[key]
        centroids.append(piece.centroidLoc)

        distances = np.linalg.norm(piece.centroidLoc - old_centroids, axis=1)
        if min(distances) > self.distance_threshold:
          new_places.append(piece.centroidLoc)
    
    board_centers_added = self.updateBoardState(y.sceneState.segIm)
    centroids = centroids + board_centers_added
    new_places = new_places + board_centers_added
    self.x.visible = len(centroids)
    
    
    centroids = np.array(centroids).reshape(-1 , 2)

    
    # OLD pieces iteration
    for piece in self.x.pieces:
      px, py = piece.location
      px, py = int(px), int(py)
      hand_occ = y.sceneState.segIm[py, px] == 150
      robot_occ = y.sceneState.tooHighMat[py, px] == 150
      # VISIBLE -> ROBOT
      # VISIBLE -> HAND
      if piece.state == trackPiece.VISIBLE:
        if hand_occ:
          piece.state = trackPiece.HAND
          piece.lastActor = trackPiece.HAND
        elif robot_occ:
          piece.state = trackPiece.ROBOT
          piece.lastActor = trackPiece.ROBOT
      elif piece.state == trackPiece.HAND or piece.state == trackPiece.ROBOT:
        # HAND -> ROBOT
        # HAND -> VISIBLE
        # HAND -> MISSING

        # ROBOT -> HAND
        # ROBOT -> VISIBLE
        # ROBOT -> MISSING
        if robot_occ:
          piece.state = trackPiece.ROBOT
          piece.lastActor = trackPiece.ROBOT
        elif hand_occ:
          piece.state = trackPiece.HAND
          piece.lastActor = trackPiece.HAND
        else:
          # Check if still in scene
          if centroids.shape[0] != 0:
            distances = np.linalg.norm(centroids - piece.location, axis=1) 
            if min(distances) > self.distance_threshold:
              piece.state = trackPiece.MISSING
              self.x.missing += 1
              # print("Piece went missing")
            else:
              piece.state = trackPiece.VISIBLE
              piece.lastActor = -1
          else:
            piece.state = trackPiece.MISSING
            self.x.missing +=1
            # print("Piece went missing")
          
      else:
        # MISSING to visible
        # Check if still in scene
        if centroids.shape[0] != 0:
          distances = np.linalg.norm(centroids - piece.location, axis=1)
          if min(distances) < self.distance_threshold:
            piece.state = trackPiece.VISIBLE
            self.x.missing -= 1
            # print("Piece came back from missing")
          elif len(new_places) > 0:
            actors = {trackPiece.HAND: "human", trackPiece.ROBOT: "robot"}
            print(f"Piece was moved by {actors.get(piece.lastActor, 'unknown')}")
            piece.location = new_places[0]
            piece.state = trackPiece.VISIBLE
            piece.lastActor = -1
            self.x.missing -= 1
            new_places.pop(0)
      

    
    # Iterate through the centroids and find center not close to
    # any of the old centers

    # Reset the first missing piece back

        


    
    
    
    
    
        
        
        
        
        
   
    
    
    
    
    
    
    
    
    
    # 
  
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
      theInstance = CoSolveActivity.loadFrom(fptr, relpath);
    else:
      theInstance = CoSolveActivity.loadFrom(fptr)

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

    theDetector = CoSolveActivity(imRegions)

    return theDetector


#
#-------------------------------------------------------------------------------
#================================ CosolveMonitor ================================
#-------------------------------------------------------------------------------
#

class CoSolveMonitor(PuzzleMonitor):
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
    actInput = CoSolveInput(sceneState=pstate, image=I)
    self.activity.process(actInput)
  
  #=============================== getInternalState ===============================
  #
  #
  def getInternalState(self):
    """!
    @brief  Get internal state of the monitor
    """

    return DetectorState(x=self.activity.x)


#============================== CosolveMonitor.py ==============================
