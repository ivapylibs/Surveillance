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

class HandIsGone(Triggers.IsNegative):
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







#
#============================= PuzzleReports =============================
