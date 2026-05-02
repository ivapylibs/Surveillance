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
import rospy
import numpy as np
from surveillance.msg import puzzPiece, puzzAction
from std_msgs.msg import Int16
from cv_bridge import CvBridge
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




#=========================== solve Trigger ===========================
class solveTrigger(Triggers.Trigger):
  """!
  @brief  Class that triggers a report when the hand leaves the
          soln zone
  """

  #======================= sortTrigger __init__ =====================
  #
  def __init__(self, theConfig = None):
    """!
    @brief  Constructor for sortTrigger trigger class
    """

    super(solveTrigger, self).__init__(theConfig)
    self.prevSig = None
    self.isInit = True
  
  #======================= sortTrigger test ==========================
  #
  def test(self, theSig):
    """!
    @brief Check if a report should be triggered for the supplied
    signal. 

    Returns true when the virtual button is pressed (rising edge)
    """

    return theSig.x.haveObs
  
#======================== solve reporter constructor ==========================
def Report_WhenPieceSolved():

  #! Trigger is when piece is sorted and hand comes out of zone
  #! Trigger is when piece to sort is dropped in zone and hand comes out of zone
  trigr = solveTrigger()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = puzzAnnouncer
  crier = Announce.Announcement(cfAnn)


   # Build the ROS channel
  channelConfigDict = dict(topic="puzz_stats", experiment='solve')
  channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  media = puzzROSChan(theConfig=channelConfig)

  theRep = Reports.Reporter(trigr, crier, media)
  return theRep
  



#=========================== sort Trigger ===========================

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
    self.isInit = True
  
  #======================= sortTrigger test ==========================
  #
  def test(self, theSig):
    """!
    @brief Check if a report should be triggered for the supplied
    signal. 

    Returns true when the virtual button is pressed (rising edge)
    """

    return theSig.x.haveObs
  
#=========================== sort announcer ===========================
def puzzAnnouncer(hsig):
  px, py = hsig.x.pcInfo.place
  px, py = int(px), int(py)
  top_left = (px - 10, py - 10)      # (x, y)
  bottom_right = (px + 10, py + 10)  # (x, y)
  color = (0, 255, 0)                # BGR for OpenCV
  thickness = 2

  # This modifies the array 'image' in-place
  if hsig.x.rgb is not None:
    img = cv2.rectangle(hsig.x.rgb.astype(np.uint8), top_left, bottom_right, color, thickness)
  else:
    img = None
  return img, hsig.x.pcInfo, hsig.x.actor


#================================= sortDepth ROS Channel ==============================

class puzzROSChan(Channel.Channel):
  """!
  @ingroup  Surveillance
  @brief    Publish the message to a topic
  """
  #==================================== init ===================================
  def __init__(self, theConfig = Channel.CfgChannel()):
    super().__init__(theConfig)
    # self.pub = rospy.Publisher(self.config.topic, puzzPiece, queue_size=1)
    self.pub = rospy.Publisher(self.config.topic, puzzAction, queue_size=1)
  #==================================== send ===================================
  #
  def send(self, data):
    img, pcInfo, actor = data
    
    # msg = puzzPiece()

    # msg.pick.x = pcInfo.pick[0]
    # msg.pick.y = pcInfo.pick[1]

    # msg.place.x = pcInfo.place[0]
    # msg.place.y = pcInfo.place[1]

    # msg.pick_time = pcInfo.pick_time
    # msg.place_time = pcInfo.place_time

    # msg.actor = actor

    # bridge = CvBridge()
    # msg.img = bridge.cv2_to_imgmsg(img, encoding="bgr8")
    # self.pub.publish(msg)

    msg = puzzAction()
    msg.loc.x = pcInfo.pick[0]
    msg.loc.y = pcInfo.pick[1]
    msg.time = pcInfo.pick_time
    msg.actor = actor
    msg.act = "pick"
    self.pub.publish(msg)

    msg = puzzAction()
    msg.loc.x = pcInfo.place[0]
    msg.loc.y = pcInfo.place[1]
    msg.time = pcInfo.place_time
    msg.actor = actor
    msg.act = "place"
    self.pub.publish(msg)

    print(f"Sent message for {actor} act")
    return True
#============================= sort Reporter Constructor ========================

def Report_WhenPieceSorted():

  #! Trigger is when piece to sort is dropped in zone and hand comes out of zone
  trigr = sortTrigger()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = puzzAnnouncer
  crier = Announce.Announcement(cfAnn)


   # Build the ROS channel
  channelConfigDict = dict(topic="puzz_stats", experiment='sort')
  channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  media = puzzROSChan(theConfig=channelConfig)

  theRep = Reports.Reporter(trigr, crier, media)
  

  return theRep


#============================= solutionProgress ROS Channel =============================

class progressTrigger(Triggers.Trigger):
  """!
  @brief  Class that triggers a report when the number of pieces
    in the solution zone changes.
  """

  #======================= progressTrigger __init__ =====================
  #
  def __init__(self, theConfig = None):
    """!
    @brief  Constructor for progressTrigger trigger class
    """

    super(progressTrigger, self).__init__(theConfig)
    self.prevSig = None
    self.isInit = True
  
  #======================= progressTrigger test ==========================
  #
  def test(self, theSig):
    """!
    @brief Check if a report should be triggered for the supplied signal.

    Returns true only when the solved-piece count increases.
    """
    if self.prevSig is None:
      self.prevSig = theSig.x.solved
      return True
    elif theSig.x.solved > self.prevSig:
      self.prevSig = theSig.x.solved
      return True
    return False

#====================== progressAnnouncer ===========================
def progressAnnouncer(hsig):
  return hsig.x.solved

#===============================  progress ROS Channel =============================

class solutionProgress(Channel.Channel):
  """!
  @ingroup  Surveillance
  @brief    Publish the message to a topic
  """
  #==================================== init ===================================
  def __init__(self, theConfig = Channel.CfgChannel()):
    super().__init__(theConfig)
    # self.pub = rospy.Publisher(self.config.topic, puzzPiece, queue_size=1)
    self.pub = rospy.Publisher(self.config.topic, Int16, queue_size=1)
  #==================================== send ===================================
  #
  def send(self, data):
    count = data
    msg = Int16()
    msg.data = count
    self.pub.publish(msg)
    print(f"Sent message for progress: {count} pieces solved")

    return True
#============================ solutionProgress Reporter Constructor =========================
def Report_SolutionProgress():

  #! Trigger is when piece to sort is dropped in zone and hand comes out of zone
  trigr = progressTrigger()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = progressAnnouncer
  crier = Announce.Announcement(cfAnn)


   # Build the ROS channel
  channelConfigDict = dict(topic="pieces_solved", experiment='cosolve')
  channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  media = solutionProgress(theConfig=channelConfig)

  theRep = Reports.Reporter(trigr, crier, media)
  

  return theRep
#
#============================= PuzzleReports =============================
