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
    lastActionTime: any = None

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
  solved     | number of pieces solved in the puzzle
  '''
  rgb : any
  pcInfo : any
  actor : str
  haveObs : bool
  solved: int

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

    self.robot_action_threshold = 10
    # Set calibrate param to false
    self.calibrated = False
   
    # Set internal state
    self.x = CoSolveState(pieces=[], visible=0, missing=0)
    
    # Set output state
    self.z = CoSolveOutput(rgb=None, pcInfo=None, actor='', haveObs=False, solved=0)

    # Set unorganized zone label
    self.unorg = 6

    # Set the solution zone label
    self.soln = 5

    # List to store all robot actions for reference during processing
    self.robotActions = []

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
  def updateBoardState(self, segIm, tooHigh):

    # Goal is to update the board state accurately
    # based on non-occluded details in the scene.

    # We need to understand what information
    # is needed by the caller about the state of the board
    # 

    occlusion_mask = np.logical_or(((self.solMask * segIm) == 150) , ((self.solMask * tooHigh) > 0))
    
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
      
      # If score is high, piece is there
      # If score is low, either occluded or no piece
      # Set missing if no occlusion, and no piece
      
      if score > threshold:
        if self.board_estimate.pieces[key].status != PieceStatus.MEASURED:
          added_centers.append(self.board_estimate.pieces[key].centroidLoc)
          self.board_estimate.pieces[key].setStatus(PieceStatus.MEASURED)
      else:
          is_occluded = np.any(occlusion_mask[rows, cols])
          if not is_occluded:
            self.board_estimate.pieces[key].setStatus(PieceStatus.GONE)
    
    # Set the pieces count output state
    self.z.solved = sum(1 for key in self.board_estimate.pieces if self.board_estimate.pieces[key].status == PieceStatus.MEASURED)
    
    return added_centers     

  #========================== rosCallback ============================
  #
  def rosCallback(self, msg):
    """!
    @brief  ROS subscriber callback to store robot action.
            Call this from your ROS subscriber.
    
    @param[in]  msg   puzzAction ROS message
    """
    if msg.actor == 'robot':
      self.robotActions.append(msg)
    # print(f"Added robot action: {msg.act} at zone {msg.zone}")

  #========================== clearRobotActions ============================
  #
  def clearRobotActions(self):
    """!
    @brief  Clear the robot actions list.
    """
    self.robotActions = []

  #============================= updateRobotActions ===================
  #
  def updateRobotActions(self, x, y, act):
    actor = 'human'
    for action in self.robotActions:
      if action.act == act:
        # Check if location matches (within threshold)
        dist = np.sqrt((action.loc.x - x)**2 + (action.loc.y - y)**2)
        if dist < self.robot_action_threshold:
          actor = 'robot'
          self.robotActions.remove(action)  # Remove matched action
          break
    return actor
  #============================== measure ==============================
  #

  def measure(self, y: CoSolveInput):
    """
    @brief  Detect human and robot activity based on scene input
    @param[in]  y  StateCosolveInput
    """
    if not self.isInit:
      return
    

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
     
      # Set the pieces in estimate
      self.updateBoardState(y.sceneState.segIm, y.sceneState.tooHighMat)
      # Add the pieces from the estimate
      for key in self.board_estimate.pieces:
        piece = self.board_estimate.pieces[key]
        if piece.status == PieceStatus.MEASURED:
          pc = trackPiece(location=piece.centroidLoc, state=trackPiece.VISIBLE)
          self.x.pieces.append(pc)
      # Update the visible piece count
      self.x.visible = len(self.x.pieces)
      
      
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
    
    board_centers_added = self.updateBoardState(y.sceneState.segIm, y.sceneState.tooHighMat)

    for key in self.board_estimate.pieces:
      piece = self.board_estimate.pieces[key]
      if piece.status == PieceStatus.MEASURED:
        centroids.append(piece.centroidLoc)

    new_places = new_places + board_centers_added
    self.x.visible = len(centroids)
    
    
    centroids = np.array(centroids).reshape(-1 , 2)

    # Filter out the robot places
    human_places = []
    robot_places = []
    for loc in new_places:
      actor = self.updateRobotActions(loc[0], loc[1], "place")
      if actor != "robot":
        human_places.append(loc)
      else:
        robot_places.append(loc)


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
              piece.lastActionTime = rospy.get_time()

              # Check attribution
              
              act = "pick"
              actor = self.updateRobotActions(px, py, act)
              if actor == 'human':
                piece.lastActor = trackPiece.HAND
              else:
                piece.lastActor = trackPiece.ROBOT
              # print("Piece went missing")
            else:
              piece.state = trackPiece.VISIBLE
              piece.lastActor = -1
          else:
            piece.state = trackPiece.MISSING
            self.x.missing +=1
            # print("Piece went missing")
            # Check attribution
            
            act = "pick"
            actor = self.updateRobotActions(px, py, act)
            if actor == 'human':
              piece.lastActor = trackPiece.HAND
            else:
              piece.lastActor = trackPiece.ROBOT
          
      else:
        # MISSING to visible
        # Check if still in scene
        if centroids.shape[0] != 0:
          distances = np.linalg.norm(centroids - piece.location, axis=1)
          if min(distances) < self.distance_threshold:
            piece.state = trackPiece.VISIBLE
            self.x.missing -= 1
            # print("Piece came back from missing")
          elif len(human_places) > 0 or len(robot_places) > 0:
            actors = {trackPiece.HAND: "human", trackPiece.ROBOT: "robot"}
            print(f"Piece was moved by {actors.get(piece.lastActor, 'unknown')}")

            actor = ''
            pick = [px, py]
            place = None
            pick_time = piece.lastActionTime
            place_time = rospy.get_time()
            if piece.lastActor == trackPiece.HAND:
              if len(human_places) == 0:
                print("Attribution error in pick")
                piece.location = robot_places[0]
                robot_places.pop(0)
              else:
                piece.location = human_places[0]
                place = human_places.pop(0)
                
                actor = 'human'
            elif piece.lastActor == trackPiece.ROBOT:
              if len(robot_places) == 0:
                print("Attribution error in pick")
                piece.location = human_places[0]
                place = human_places.pop(0)
                actor = 'human'
              else:
                piece.location = robot_places[0]
                robot_places.pop(0)
            piece.state = trackPiece.VISIBLE
            piece.lastActor = -1
            self.x.missing -= 1
            if actor == 'human':
              self.z.haveObs = True
              self.z.actor = actor
              self.z.pcInfo = Piece(
                  pick=pick,
                  place=place,
                  pick_time=pick_time,
                  place_time=place_time,
                  zone = -1
              )
      

    
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
    # Reset so that reporter ignores it
    self.z.haveObs = False
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
  
  @param[in]  theParams   Configuration parameters for the monitor.
  @param[in]  thePerceiver  The perceiver whose state is being monitored.
  @param[in]  theActivity    The activity recognizer whose output is being monitored.
  @param[in]  theReporters    The reporter to use for generating reports.

  '''
  def __init__(self, theParams, thePerceiver, theActivity, theReporters=None):
    super().__init__(theParams, thePerceiver, theActivity, None)

    self.reporters = theReporters


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
  
  #=============================== process ===============================
  #
  #
  def process(self, I):
    """!
    @brief  Run perceive + activity recognize pipeline for one step/image
            measurement. Run the output through the reporters
    """

    self.predict()
    self.measure(I)
    self.correct()
    self.adapt()

    if self.reporters is not None:
      for reporter in self.reporters:
        print("Processing reporter")
        reporter.process(self.getState())

  #=============================== getInternalState ===============================
  #
  #
  def getInternalState(self):
    """!
    @brief  Get internal state of the monitor
    """

    return DetectorState(x=self.activity.x)


#============================== CosolveMonitor.py ==============================
