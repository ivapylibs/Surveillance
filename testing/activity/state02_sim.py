"""

    @brief         Test the extraction of the human hand states from the simulated data

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           02/13/2022

"""

import numpy as np
import warnings
from skimage import draw
import matplotlib.pyplot as plt

import Lie.group.SE2.Homog
import improcessor.basic as improcessor
import operator
import detector.inImage as detector
import trackpointer.centroid as tracker
from Surveillance.activities.state import StateEstimator

# Borrow the moving triangle simulation code from the trackpointers test file
class fakeTriangle(object):

  def __init__(self, pMark=None, sMark=None, imSize=None, gamma=None):

    self.g = Lie.group.SE2.Homog()
    self.pMark = pMark
    self.isDirty = True

    if sMark is None:
      self.noMarker = True
    elif sMark.ndim!=3: 
      if pMark.shape[1]>1:
        # We put channel in the first dimension
        self.sMark = np.repeat(sMark[:, :, np.newaxis], pMark.shape[1], axis=2).transpose(2,0,1)
      else:
        self.sMark = sMark
      self.noMarker = False
    elif len(sMark) == pMark.shape[1]:
      self.sMark = sMark
      self.noMarker = False
    else:
      warnings.warn('fakeTriangle: pMark and sMark are incompatible sizes.')

    self.imSize = imSize

    if gamma and np.isscalar(gamma):
      self.gamma = gamma

  def setPose(self, gSet):

    self.g = gSet
    self.isDirty = True

  def render(self):

    if self.isDirty:
      self.I = self.synthesize()

    I = self.I
    return I


  def synthesize(self):

    if (not self.isDirty):    #! Check just in case, to avoid recomputing.
      return

    sW = []
    #! Step 1: Map marker centers to new locations. (maybe not be needed)
    pW = self.g * self.pMark # 3*N

    image_shape = (self.imSize[0], self.imSize[1])

    #! Step 2: Map shape (polygon) to new locations.
    if self.noMarker:
      I = draw.polygon2mask(image_shape, pW[:2,:].T)
    else:
      for ii in range(len(self.sMark)):
        sW.append(self.g * (self.pMark[:,ii].reshape(-1,1)) + self.sMark[ii]) # 3*N

      I = np.zeros(self.imSize).astype('bool')
      for ii in range(len(sW)):
        I = I | draw.polygon2mask(image_shape, sW[ii][:2,:].T)

    return I.astype('uint8')


if __name__ == "__main__":

    #==[1] The cute simulator
    pMark  = np.array([[-12, -12, 12],[18, -18, 0],[1, 1, 1]])+np.array([[120],[120],[0]]) # Define a triangle
    sMark  = 2*np.array([[ -2, -2, 2, 2],[-2, 2, 2, -2],[0, 0, 0, 0]]) # For each vertice
    imSize = np.array([200, 200])

    useMarkers = False
    # useMarkers = True

    if useMarkers:
      ftarg = fakeTriangle(pMark, sMark, imSize)
    else:
      ftarg = fakeTriangle(pMark, None, imSize)


    #==[2] Detector and the trackpointer

    improc = improcessor.basic(operator.gt,(0,),
                               improcessor.basic.to_uint8,())
    binDet = detector.inImage(improc)
    trackptr = tracker.centroid()

    #==[3] The state parser
    state_parser = StateEstimator(
        signal_number=3,
        signal_names=["location", "Foo", "Foo2"],
        state_number=1,
        state_names=["Move"],
        move_th=2
    ) 


    #==[4]  start simulation 

    # init
    theta = 0
    R = Lie.group.SE2.Homog.rotationMatrix(theta)
    x = np.array([[200],[200]])
    g = Lie.group.SE2.Homog(R=R, x=x)
    ftarg.setPose(g)

    plt.ion()

    # Render first frame @ initial pose.
    I = ftarg.render()
    fh_sim = plt.figure(1)
    plt.imshow(I, cmap='Greys')

    # random pause-related
    pause_count = -1
    pause_count_th = 20     # Pause for 20 frames each time
    pause_prob = 0.2

    # start running
    for ii in range(1000):# Loop to udpate pose and re-render.
        
        # determine the pause
        # if already pause or
        if (pause_count >= 0 and pause_count < pause_count_th) or (np.random.rand() < pause_prob):
            theta = 0
            pause_count = pause_count + 1
            # if reach the threshold, reset
            if pause_count >= pause_count_th:
                pause_count = -1
        else:
            theta = np.pi/100

        # render image
        R = Lie.group.SE2.Homog.rotationMatrix(theta)
        g = g * Lie.group.SE2.Homog(x=np.array([[0],[0]]), R=R)
        ftarg.setPose(g)
        I = ftarg.render()

        # get the trackpointer location
        binDet.process(I)
        dI = binDet.Ip
        trackptr.process(dI)
        tstate = trackptr.getState()
        if ii==0:
            # Start tracking
            trackptr.setState(tstate)

        # parse the state
        state_parser.process([tstate.tpt, None, None])
        
        # display
        plt.figure(1)
        plt.cla()
        trackptr.displayState()
        plt.imshow(I, cmap='Greys')

        state_parser.visualize_state_evolving()

        plt.pause(0.001)

    plt.ioff()
    plt.draw()


    #
    #=============================== trackTri01 ==============================

