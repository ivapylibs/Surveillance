#============================= PuzzleReports =============================
##
# @package  Surveillance.layers.PuzzleReports
#
# @brief    Collection of classes for reporting out statistics related to
#           puzzle related tasks.
#
# The basic reporters assume that a simple signal is being sent in.  If the
# system is passing along a nontrivial signal, then more work is needed.
# That may involve using a pre-processor or just overloading a given scheme
# to first generate the necessary input signal from the passed state signal.
# This package takes care of that.
#
# @ingroup  Surveillance
#
# @author   Nihit Agarwal       nagarwal90@gatech.edu
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2026/02/11
#
#============================= PuzzleReports =============================

# Specify imports
import perceiver.reports.drafts   as Announce
import perceiver.reports.triggers as Triggers
import perceiver.reports.channels as Channel
import perceiver.reporting        as Reports
import cv2
import os
import time
import rospy
import numpy as np
from mary_interface.msg import pickPlace
# from std_msgs.msg import Header

#====================== HandIsGone reporting ===============================

#===================== HandIsGone Trigger =================================
class HandIsGone(Triggers.Trigger):
  def __init__(self, theConfig = None):
    """!
    @brief  Constructor for hand is gone trigger.

    @param[in]  theConfig   Configuration specs (optional).
    """
    #if (theConfig is None):
    #  theConfig = CfgTrigger()
    super(HandIsGone,self).__init__(theConfig)

  def test(self, actSig):
    """!
    @brief  Snag the hand activity state part of the signal for testing.

    @param[in]  actSig  The activity state signal.
    """
    return super(HandIsGone,self).test(actSig.x.hand[0])

#========================= Announcer Func ===============================
def HandStatus(hsig):
  #hstr = str(hsig.x.hand[0])
  zstr = str(hsig.x.getZoneCounts())
  return zstr

#======================= Reporter constructor ==========================
def Report_WhenHandIsGone():

  #! Trigger is when hand leaves the scene. Then we can see "everything."
  trigr = HandIsGone()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = HandStatus
  crier = Announce.Announcement(cfAnn)

  #! Next build the channel.
  media = Channel.Channel()

  theRep = Reports.Reporter(trigr, crier, media)
  return theRep


#=========================== sort reporting =========================

#=========================== sort Trigger ===========================
class sortTrigger(Triggers.Trigger):
  """!
  @brief  Class that triggers a report when the hand leaves a zone
          and the number of pieces in a zone increases
  """

  #======================= sortTrigger __init__ =====================
  #
  def __init__(self, theConfig = None):
    """!
    @brief  Constructor for sortTrigger trigger class
    """

    super(sortTrigger, self).__init__(theConfig)
    self.prevSig = None
    self.isInit = False
  
  #======================= sortTrigger test ==========================
  #
  def test(self, theSig):
    """!
    @brief Check if a report should be triggered for the supplied
    signal. 

    Returns true when the virtual button is pressed (rising edge)
    """

    pieceSorted = False
    if self.isInit:
      if not self.prevSig.x.btnPressed and theSig.x.btnPressed and theSig.x.pcInfo is not None:
        pieceSorted = True
    else:
      self.isInit = True
    
    # Set state of the trigger
    self.prevSig = theSig
    # if pieceSorted:
    #   print("actually detected")
    return pieceSorted
  

#================================== sort Announcer time ============================
def getTimestamp(hsig):
  # Compute average time
  t = time.process_time()
  print(hsig.x.pcInfo.zone)
  return [hsig.x.pcInfo.pick, hsig.x.pcInfo.place, str(t)]

#================================== sort Announcer img ============================

def getImage(hsig):
  px, py = hsig.x.pcInfo.place
  px, py = int(px), int(py)
  top_left = (px - 10, py - 10)      # (x, y)
  bottom_right = (px + 10, py + 10)  # (x, y)
  color = (0, 255, 0)                # BGR for OpenCV
  thickness = 2

  # This modifies the array 'image' in-place
 
  img = cv2.rectangle(hsig.x.image.astype(np.uint8), top_left, bottom_right, color, thickness)
  return img

#================================= sort Time Channel ==============================

class puzzTimeChannel(Channel.toCSV):
  """!
  @ingroup  Surveillance
  @brief    Save the announcement to csv
  """

  #==================================== send ===================================
  #
  def send(self, theRow):
    if theRow is None:
      print("Skipping")
      return False

    if self.config.runner is not None:
      self.config.runner += 1
      outRow = [self.config.runner]
      outRow.extend(theRow)
      self.writer.writerow(outRow)
      # @todo see if writerow returns success status?
    else:
      self.writer.writerow(theRow)

    return True

#================================= sort Img Channel ==============================
class puzzImgChannel(Channel.Channel):
  """!
  @ingroup  Surveillance
  @brief    Save the signal (image) to a folder
  """
  #==================================== init ===================================
  #
  def __init__(self, theConfig = Channel.CfgChannel()):
    super().__init__(theConfig)
  #==================================== send ===================================
  #
  def send(self, image):
    if image is None:
      print("Skipping")
      return False

    if self.config.runner is not None:
      self.config.runner += 1
    else:
      self.config.runner = 1
    # Save the image
    filePath = os.path.join(self.config.image_dir, f"{self.config.experiment}_{self.config.runner}.png")
    # print("Inside channel, type of image is ", type(image))
    image = image[::-1, ::-1] # rotate 180 deg

    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filePath, bgr_image)

    
    return True


#================================= sort ROS Channel ==============================

class puzzROS(Channel.Channel):
  """!
  @ingroup  Surveillance
  @brief    Publish the message to a topic
  """

  #==================================== send ===================================
  #
  def send(self, data):
    pub = rospy.Publisher(self.config.topic, pickPlace, queue_size=1)
    msg = pickPlace()

    # msg.header = Header(stamp=rospy.Time.now())
    msg.pick.x = data[0][0]
    msg.pick.y = data[0][1]

    msg.place.x = data[1][0]
    msg.place.y = data[1][1]

    pub.publish(msg)
    return True
#============================= sort Reporter Constructor ========================

def Report_WhenPieceSorted():

  #! Trigger is when piece to sort is dropped in zone and hand comes out of zone
  trigr = sortTrigger()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = getTimestamp
  crier = Announce.Announcement(cfAnn)

  #! Next build the channel.
  # channelConfigDict = dict(end="\n", filename= "data/sortReport.csv", experiment='sort', otype="w")
  # channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  # media = puzzTimeChannel(theConfig=channelConfig)
  # media.setRunner(0)

  # Build the ROS channel
  channelConfigDict = dict(topic="human_sort_time", experiment='sort')
  channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  media = puzzROS(theConfig=channelConfig)


  #! Create the reporter
  theTimeRep = Reports.Reporter(trigr, crier, media)

  #! Trigger is when accuacy of sort is needed as soon as hand leaves workspace,
  #  image of sort zones is captured
  trigr = sortTrigger()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = getImage
  crier = Announce.Announcement(cfAnn)

  #! Next build the channel.
  channelConfigDict = dict(end="\n",image_dir='images', experiment='sort', otype="w", runner=0)
  channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  media = puzzImgChannel(theConfig=channelConfig)

  #! Create the reporter
  theImgRep = Reports.Reporter(trigr, crier, media)

  return theTimeRep, theImgRep



#================================= solve Reporting ==============================

#================================= solve Configs ==============================
#
class CfgSolveTrigger(Triggers.CfgTrigger):
  """!
  @brief  Configuration instance for Solve Trigger
  """
  #------------------------------ __init__ -----------------------------
  #
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):
    '''!
    @brief    Instantiate a trigger build configuration.
    '''

    if init_dict is None:
      init_dict = CfgSolveTrigger.get_default_settings()

    super(CfgSolveTrigger,self).__init__(init_dict, key_list, new_allowed)


  #------------------------ get_default_settings -----------------------
  #
  @staticmethod
  def get_default_settings():
    """!
    @brief  Get default build configuration settings for Trigger.
    """

    default_settings = dict(threshold=100)
    return default_settings

#================================= solve Trigger ==============================
#
class solveTrigger(Triggers.Trigger):
  """!
  Class that triggers a report when hand leaves puzzle solution 
  region and are of pieces in solution region increases
  """
  #======================= solveTrigger __init__ =====================
  #
  def __init__(self, theConfig = None):
    """!
    @brief  Constructor for solve trigger class
    """

    if (theConfig is None):
      theConfig = CfgSolveTrigger()

    self.config = theConfig
    self.prevSig = None
    self.old_area = None
    self.isInit = False

#================================= solve drop trigger ==============================
#
class solveDropTrigger(solveTrigger):
  
  """!
  @brief Class that helps detect when a piece was dropped into
         solution zone by checking when hand leaves the zone.
  """
  #======================= solveTrigger test ==========================
  #
  def test(self, theSig):
    """!
    @brief Check if a report should be triggered for the supplied
    signal. 

    Compare the passed signal from the last check (if there was one)
    and return True if hand left a puzzle board and number of pieces 
    in puzzle solution increased. 

    On startup, there may be no previous signal.  In that case the first invocation
    returns a False and stores the signal for future invocations.
    """

    pieceSolved = False
    if self.isInit:
      # compute the location of hand in old signal
      old_hand_loc = self.prevSig.x.hand[0]

      # compute the location of hand in new signal
      new_hand_loc = theSig.x.hand[0]

      if new_hand_loc < 0 and old_hand_loc >= 0: # hand left a zone
        # compute the pc count in old signal
        # print("Hand left zone")
        old_area = self.old_area

        # compute the pc count in new signal
        new_area = theSig.x.area
        # print(f"Old area: {old_area} and new area {new_area}")
        # print("old count: ", old_pc_count)
        # print("new pc count: ", new_pc_count)
        if new_area - old_area > self.config.threshold:
          self.old_area = new_area
          pieceSolved = True
          # print("write to csv")
    else:
      self.isInit = True
      self.old_area = theSig.x.area
    self.prevSig = theSig
   
    return pieceSolved
  
#======================== solve reporter constructor ==========================

def Report_WhenPieceSolved():

  #! Trigger is when piece is sorted and hand comes out of zone
  trigr = solveTrigger()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = signalParser
  crier = Announce.Announcement(cfAnn)

  #! Next build the channel.
  channelConfigDict = dict(end="\n", filename= "data/solveReport.csv",\
                            image_dir="images", experiment='solve', otype="w")
  channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  media = puzzleChannel(theConfig=channelConfig)
  media.setRunner(0)

  theRep = Reports.Reporter(trigr, crier, media)
  return theRep


#
#============================= PuzzleReports =============================
