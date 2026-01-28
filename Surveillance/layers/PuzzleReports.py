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
# @author   Patricio A. Vela,   pvela@gatech.edu
# @date     2025/07/31
#
#============================= PuzzleReports =============================

import perceiver.reports.drafts   as Announce
import perceiver.reports.triggers as Triggers
import perceiver.reports.channels as Channel
import perceiver.reporting        as Reports
import cv2
import os

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

def HandStatus(hsig):
    #hstr = str(hsig.x.hand[0])
    zstr = str(hsig.x.getZoneCounts())
    return zstr

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

#####################################################################
# Trigger for detecting when the hand leaves a zone and the number
# of pieces increases in any zone. 



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
    self.old_pc_count = None
    self.isInit = False
  
  #======================= sortTrigger test ==========================
  #
  def test(self, theSig):
    """!
    @brief Check if a report should be triggered for the supplied
    signal. 

    Compare the passed signal from the last check (if there was one)
    and return True if hand left a zone and number of pieces in zones 
    increased. 

    On startup, there may be no previous signal.  In that case the first invocation
    returns a False and stores the signal for future invocations.
    """

    pieceSorted = False
    if self.isInit:
      # compute the location of hand in old signal
      old_hand_loc = self.prevSig.x.hand[0]

      
      # compute the location of hand in new signal
      new_hand_loc = theSig.x.hand[0]

      if new_hand_loc <= 0 and old_hand_loc > 0: # hand left a zone
        # compute the pc count in old signal
        print("Hand left zone")
        old_pc_count = self.old_pc_count

        # compute the pc count in new signal
        new_pc_count = theSig.x.getZoneCounts()
        # print("old count: ", old_pc_count)
        # print("new pc count: ", new_pc_count)
        for i in range(1, len(old_pc_count)):
          if new_pc_count[i] > old_pc_count[i]:
            pieceSorted = True
            break
        self.old_pc_count = new_pc_count
    else:
      self.isInit = True
      self.old_pc_count = theSig.x.getZoneCounts()
    self.prevSig = theSig
    # if pieceSorted:
      # print("actually detected")
    return pieceSorted

######################################################################################
# Formatting function for announcer to extract useful information from signal
# and to package it with meta information
######################################################################################

#################################### signalParser ####################################
def signalParser(hsig):
    from datetime import datetime
    now = datetime.now()
    return hsig.x.image, [str(now.time())]

#####################################################################################
# Custom channel to help write to csv file along with save the image file
#####################################################################################

############################## sortChannel ########################################
class sortChannel(Channel.toCSV):
  """!
  @ingroup  Surveillance
  @brief    Save the announcement to csv and signal (image) to a folder
  """

  #==================================== send ===================================
  #
  def send(self, announcement):
    image, theRow = announcement
    if theRow is None:
      print("Skipping")
      return False

    if self.config.runner is not None:
      outRow = [self.config.runner]
      outRow.extend(theRow)
      self.writer.writerow(outRow)
      # @todo see if writerow returns success status?
    else:
      self.writer.writerow(theRow)
    
    # Save the image
    filePath = os.path.join(self.config.image_dir, f"sorted_{self.config.runner}.png")
    print("Inside channel, type of image is ", type(image))
    cv2.imwrite(filePath, image.color)

    return True
  
#################################################################################
# Function to generate the reporter
#################################################################################
def Report_WhenPieceSorted():

  #! Trigger is when piece is sorted and hand comes out of zone
  trigr = sortTrigger()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = signalParser
  crier = Announce.Announcement(cfAnn)

  #! Next build the channel.
  channelConfigDict = dict(end="\n", filename= "solveReport.csv", image_dir="images", otype="w")
  channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  media = sortChannel(theConfig=channelConfig)
  media.setRunner(0)

  theRep = Reports.Reporter(trigr, crier, media)
  return theRep



#========================================================================
# Trigger for detecting when area inside the puzzle region increases
# significantly, indicating a piece was added (done only when hand )
# is not in the puzzle region.

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
        print("Hand left zone")
        old_area = self.old_area

        # compute the pc count in new signal
        new_area = theSig.x.area
        print(f"Old area: {old_area} and new area {new_area}")
        # print("old count: ", old_pc_count)
        # print("new pc count: ", new_pc_count)
        if new_area - old_area > self.config.threshold:
          self.old_area = new_area
          pieceSolved = True
          print("write to csv")
    else:
      self.isInit = True
      self.old_area = theSig.x.area
    self.prevSig = theSig
   
    return pieceSolved

def fixedFun(igsig=None):
    from datetime import datetime
    now = datetime.now()
    return [str(now.time())]

def Report_WhenPieceSolved():

  #! Trigger is when piece is solved and hand comes out of zone
  trigr = solveTrigger()

  #! Define the announcement type first.
  cfAnn = Announce.CfgAnnouncement()
  cfAnn.signal2text = fixedFun
  crier = Announce.Announcement(cfAnn)

  #! Next build the channel.
  channelConfigDict = dict(end="\n", filename= "solveReport.csv", otype="w")
  channelConfig = Channel.CfgChannel(init_dict=channelConfigDict)
  media = Channel.toCSV(theConfig=channelConfig)
  media.setRunner(0)

  theRep = Reports.Reporter(trigr, crier, media)
  return theRep


#
#============================= PuzzleReports =============================
