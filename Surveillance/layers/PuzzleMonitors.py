#============================== PuzzleMonitors ==============================
##
# @package  Surveillance.layers.PuzzleMonitors
#
# @brief    Monitor systems with activity classes for monitoring human 
#           sort and solve.
#
# @ingroup  Surveillance
#
# @author   Nihit Agarwal,   nagarwal90@gatech.edu
# @date     2026/02/11
#
#============================== PuzzleMonitors ==============================

#--[0.A] Standard python libraries.
#
import numpy as np
import scipy
from dataclasses import dataclass
import h5py
from skimage.segmentation import watershed
from skimage import morphology
from Surveillance.layers.PuzzleScene import *
import matplotlib.pyplot as plt
import time


#============================= Sort Monitoring ==============================
#

@dataclass
class Piece:
  '''!
  @ingroup Surveillance
  @brief   Piece information is defined here. Captures pick, drop, and zone of
           drop.
  
  Contents of the dataclass are:
  field    |  description
  ---------|--------------
  pick     | pick coordinates in pixel space
  place    | place coordinates in pixel space
  zone     | zone where piece was dropped (if sorting)
  '''
  pick: np.ndarray
  place: np.ndarray
  zone: any


@dataclass
class StateSortActivity:
  '''!
  @ingroup  Surveillance
  @brief    Sort activity state is defined here. Captures current state
            of hand in the sort process.

  Contents of the dataclass are:
  field    |   description
  ---------|-----------------
  hand     | zone that hand is in currently
  image    | RGB image
  segIm    | ndarray representing the segmented image
  zones    | dictionary mapping each zone to # of pieces in it
  btnPressed | is the hand over the button
  pcInfo     | piece information


  '''
  hand      : int
  image     : any
  segIm     : any
  zones     : ZoneCounter
  btnPressed : bool
  pcInfo     : Piece

  #=========================== getZoneCounts ===========================
  #
  #
  def getZoneCounts(self):
    zsort = sorted(self.zones.keys())
    return [self.zones[z] for z in zsort]


@dataclass
class SortActivityInput():
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
  sceneState: any
  image: any

#
#-----------------------------------------------------------------------------
#================================ SortActivity ==============================
#-----------------------------------------------------------------------------
#

class SortActivity(PuzzleActivities):
  """!
    @brief  Class detects the state of the puzzle sort region.
  """

  #============================== __init__ ===============================
  #
  def __init__(self, imRegions):
    super().__init__(imRegions)
    self.zonemask = ((self.imRegions > 0) & (self.imRegions != 5)).astype(int)
    self.start = True
  
  #=============================== measure ===============================
  #
  def measure(self, y):
    """
    @brief  Checks the measurment of hand and board to determine sort
            state.
    @param[in]  y  SortActivityInput
    """
    
    noCounts = ZoneCounter({x:0 for x in range(self.lMax+1)}) # Takes care of case where no piece in a zone
    if self.z is None:
      currState = StateSortActivity(hand=-1, image=None, segIm=None, zones=None, btnPressed=False, pcInfo=None)
    else:
      currState = StateSortActivity(hand=-1, image=None, segIm=self.z.segIm, zones=None, btnPressed=False, pcInfo=None)
   
    if not self.isInit: # probably checks for the zone calibration
      return


    # Check if hand transitioned from a zone to another 
    if y.sceneState.isHand: 
      # Map coordinates takes in (i,j). Map zsig from (x,y) to (i,j).
      yhand  = np.flipud(y.sceneState.hand)
      currState.hand = scipy.ndimage.map_coordinates(self.imRegions, yhand, order = 0)[0]
    else:
      currState.hand = -1
    
   
    # Count pieces in each zone
    if y.sceneState.isPuzzle:
      ypuzz  = np.flipud(y.sceneState.puzzle)
      puzzle  = scipy.ndimage.map_coordinates(self.imRegions, ypuzz, order = 0)
      
      currState.zones = ZoneCounter(puzzle) + noCounts
     
    
    

    # Run state machine logic
    # Set button pressed signal and capture image, segmented Image
    if currState.hand == 5:
      currState.btnPressed = True
      
    # Detect transition to button
    if self.start or (currState.hand == 5 and self.z.hand != 5):
      # Set capture image, segmented Image at button press time
      currState.image = y.image.color
      currState.segIm = y.sceneState.segIm * self.zonemask

      if not self.start:
        # Compute piece details
        # print(f"currState segIm: {currState.segIm}")
        diff = currState.segIm - self.z.segIm
        diff[diff  == -75] = 150

        output_diff = np.zeros_like(diff)
        labels = [75, 150]
        for val in labels:
          # Create a binary mask just for this label
          mask = (diff == val)
          clean_mask = morphology.remove_small_objects(mask, min_size=10)
          output_diff[clean_mask] = val

        import matplotlib.pyplot as plt
        plt.imshow(output_diff)
        
        plt.show()

        # find pick spot
        indices_pick = np.where(output_diff == 150)
        if len(indices_pick[0]) > 0:
          pick_y = np.mean(indices_pick[0])
          pick_x = np.mean(indices_pick[1])
        
        # find drop spot
        indices_place = np.where(output_diff == 75)
        if len(indices_place[0]) > 0:
          place_y = np.mean(indices_place[0])
          place_x = np.mean(indices_place[1])
        
        # Find drop zone
        if len(indices_pick[0]) > 0 and len(indices_place[0]) > 0:
          # print(f"{place_x}, {place_y} are place coord")
          zone_drop =  self.imRegions[int(place_y), int(place_x)]
          currState.pcInfo = Piece(pick=np.array([pick_x, pick_y]), place=np.array([place_x, place_y]), zone=zone_drop)
        else:
          print("Did not find piece pick - drop")
      else:
        print("Captured first segIm, setting start to false")
        self.start = False
      
    # Set new detector / activity state
    self.z = currState
      


  
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
      theInstance = SortActivity.loadFrom(fptr, relpath);
    else:
      theInstance = SortActivity.loadFrom(fptr)

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

    theDetector = SortActivity(imRegions)

    return theDetector


#================================ sort activity v2 =============================
#

@dataclass
class SortState:
  '''!
  @ingroup  Surveillance
  @brief    Sort activity state is defined here. Captures current state
            of hand in the sort process.

  Contents of the dataclass are:
  field    |   description
  ---------|-----------------
  handZones| list that tracks if hand is in the zone or not
  handTimeZones | list that tracks the last time hand was in that zone

  '''
  handZones: list
  handTimeZones: list


class SortActivityDepth(PuzzleActivities):
  '''!
    @brief  Detects sort activities by checking depth and color across the zone
            boundaries
  '''

  #============================== __init__ ===============================
  #
  def __init__(self, imRegions):
    super().__init__(imRegions)

    # Scan zones and store the boundary indices into a list

    # attempt to create the boundary zone markers and display

    self.zone_boundaries = [None] * (self.lMax + 1)

    def get_precise_boundary(arr, label_value=1):
      # 1. Isolate the blob
      mask = (arr == label_value).astype(np.uint8)
      
      contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

      if contours:
        # contours[0] is an array of shape (N, 1, 2) containing [[x, y], ...]
        boundary_points = contours[0].reshape(-1, 2) 
      return boundary_points
    
    for i in range(1, self.lMax + 1):
      b_ind = get_precise_boundary(self.imRegions, i)
      self.zone_boundaries[i] = b_ind
    
    # Set calibrate param to false
    self.calibrated_depth = False
    self.depth_threshold = 0.05



    
  #=============================== measure ===============================
  #
  def measure(self, y):
    """
    @brief  Checks the measurment of hand and board to determine sort
            state.
    @param[in]  y  SortActivityInput
    """
    if not self.isInit:
      return
    
    self.z = SortState(handZones=[0] * (self.lMax + 1), handTimeZones=[0] * (self.lMax + 1))

    # Run a depth scan on the image boundaries of each zone
    # Check for high values on the boundaries, and 
    # set the state based on that
    dep = y.image.depth

    zone_boundary_depths = [None] * (self.lMax + 1)
    for i in range(1, self.lMax + 1):
      b_ind = self.zone_boundaries[i]
      x_coords = b_ind[:, 0]
      y_coords = b_ind[:, 1]

      boundary_depths = dep[y_coords, x_coords]

      zone_boundary_depths[i] = boundary_depths
    
    if not self.calibrated_depth:
      self.calibrated_depth = True
      self.zone_boundary_depths = zone_boundary_depths
    else:
      # Check the differences in the zone boundary depths
      for i in range(1, self.lMax + 1):
        diff = self.zone_boundary_depths[i] - zone_boundary_depths[i]
        if np.max(diff) > self.depth_threshold:
          self.z.handZones[i] = 1
          self.z.handTimeZones[i] = time.process_time()
      

    # print(f"Minimum depth for zone {1} is {min_depth[1]}")





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
      theInstance = SortActivityDepth.loadFrom(fptr, relpath);
    else:
      theInstance = SortActivityDepth.loadFrom(fptr)

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

    theDetector = SortActivityDepth(imRegions)

    return theDetector

#
#-------------------------------------------------------------------------------
#================================ SortMonitor ================================
#-------------------------------------------------------------------------------
#

class SortMonitor(PuzzleMonitor):
  '''!
  @ingroup  Surveillance
  @brief    Sort monitor that examines the workscene perceiver state and
            helps generate sort reports.

  '''

  #============================ SortMonitor ============================
  #
  #
  def __init__(self, theParams, thePerceiver, theActivity, theReporter = None, theSecondReporter=None):
    """!
    @brief  Constructor for the perceiver.monitor class.
  
    @param[in] theParams    Option set of paramters. 
    @param[in] thePerceiver Perceiver instance (or possibly not).
    @param[in] theActivity  Activity detector/recognizer.
    @param[in] theReporter  Reporting mechanism for activity outputs.
    """

    super(SortMonitor,self).__init__(theParams, thePerceiver, theActivity, theReporter)
    self.secondReporter = theSecondReporter

  def measure(self, I):
    """!
    @brief  Run activity detection process to generate activity state measurement. 
            If perceiver has no measurement/observation, then does nothing.

    @param[in]  I   Image to process. Depending on implementation, might be optional.
    """
    if not self.params.external:    # Perceiver process not externally called.
      self.perceiver.process(I)     # so should run perceiver process now.

    pstate = self.perceiver.getState()
    actInput = SortActivityInput(sceneState=pstate, image=I)
    self.activity.process(actInput)

  #=============================== process ===============================
  #
  #
  def process(self, I):
    """!
    @brief  Run perceive + activity recognize pipeline for one step/image
            measurement.
    """

    super().process(I)
    if self.secondReporter is not None:
      self.secondReporter.process(self.getState())


#================================== Solve Monitoring =========================
@dataclass
class StateSolveActivity():
  '''!
  @ingroup  Surveillance
  @brief    Perceiver puzzle scene activity state structure for solving
            by human. 

  Contents of this dataclass are:
  field    | description
  -------- | -----------
  area     | Area of pices in puzzle
  hand     | state of the hand
  image    | raw image from camera
  '''
  
  area     : int
  hand     : any
  image    : any

@dataclass
class SolveActivityInput():
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
  sceneState: any
  image: any

#
#-----------------------------------------------------------------------------
#================================ SolveActivity ==============================
#-----------------------------------------------------------------------------
#

class SolveActivity(PuzzleActivities):
  """!
  @brief  Class detects the state of the puzzle solution region and in 
          terms area signal
  """

  #============================== measure ==============================
  #
  def measure(self, y):
    """
    @brief  Compare image signal to empty solution board state.
    @param[in]  y  PuzzleScene state
    """
    puzzleSeg = self.imRegions * y.sceneState.segIm
    # print(np.unique(puzzleSeg))
    self.z = StateSolveActivity(0, [-1], None)
    self.z.area = np.count_nonzero(puzzleSeg)

    if y.sceneState.isHand: 
      # Map coordinates takes in (i,j). Map zsig from (x,y) to (i,j).
      yhand  = np.flipud(y.sceneState.hand)
      self.z.hand = scipy.ndimage.map_coordinates(self.imRegions, yhand, order = 0)
    else:
      self.z.hand = [-1]
    
    self.z.image = y.image.color
  
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
#================================ SolveMonitor ================================
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
    actInput = SolveActivityInput(sceneState=pstate, image=I)
    self.activity.process(actInput)

#
#============================== PuzzleMonitors ==============================
