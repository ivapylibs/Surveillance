#====================== Surveillance.deployment.Monitor =====================
"""
@ brief:    The Monitor class for the Surveillance system.

@author:    Nimisha Pabbichetty, npabbichetty3@gatech.edu
            Yiye Chen,           yychen2019@gatech.edu
            Yunzhi Lin,          yunzhi.lin@gatech.edu
@date:      

"""
#====================== Surveillance.deployment.Monitor =====================
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
import cv2
import numpy as np

from dataclasses import dataclass
from benedict import benedict

from puzzle.runner import RealSolver
from Surveillance.activity.state import StateEstimator
from camera.utils.display import display_images_cv
from Surveillance.activity.utils import DynamicDisplay, ParamDynamicDisplay
from puzzle.piece.template import PieceStatus


@dataclass
class Params:
    #TODOL create YAML
    #list of topics -> use in ROS Wrapper instead, still include in YAML tho, create 2 YAMLs and merge in wrapper, one basic for monitor and one with ROS stuff
    fDir: str = "./"
    rosbag_name: str = "testing/data/tangled_1_work.bag"
    real_time: bool = False

    puzzle_solver_SolBoard: str = "testing/data/caliSolBoard.obj"
 

    #=========================== __contains__ ==========================
    """
    @brief  Overload the "in" operator to request whether the class has
            the targeted member variable.
    
    @param[in]  att_name    The member variable (attribute name).
    
    """
    def __contains__(self, att_name):
        return hasattr(self, att_name)


    #========================== set_from_dict ==========================
    """
    @brief  Overwrite default parameters from dictionary. 
            Only if in dictionary and in dataclass instance will they be
            set. Otherwise, not set.

    @param[in]  pdict   The dictionary of parameter settings.
    """
    def set_from_dict(self, pdict):

        for key in pdict.items:
            if key in self.params:
                setattr(self.params, key, getattr(pdict,key))


    #========================== set_from_yaml ==========================
    """
    @brief  Overwrite default parameters from a yaml file specification.
            Only if in yaml file and in dataclass instance will they be
            set. Otherwise, not set.

    @param[in]  yfile   The yaml file with parameter settings.
    """
    def set_from_yaml(self, yfile):

        ydict = benedict.from_yaml(yfile)    # load yaml file.
        self.set_from_dict(ydict)

#========================== Class: SurveillanceMonitor==========================
'''plt.ion()
fig=plt.figure()
ax = fig.add_subplot(1, 1, 1)

def plotProgress(i,x,y):
        ax.clear()               
        ax.plot(x,y)'''

class SurveillanceMonitor():
    def __init__(self, puzzle_solver: RealSolver(), state_parser: StateEstimator(signal_number=1), params: Params = Params()): #get the puzzle_solver object
        
        self.params=params
        self.puzzle_solver=puzzle_solver
        self.state_parser=state_parser

        # Fig for puzzle piece status display
        self.status_window = None
        self.activity_window = None
        self.progress_window = None

        self.puzzle_solver_mode=0
        self.callback_id=0
        self.ros_pub = False

        self.plan=None
        self.activity_data=None

        self.progress_tracking=dict()
        plt.ion()
        
    
    def HandStateAnalysis(self,hTracker,rgb):
        self.move_state = -1
        if hTracker is None:
                    # -1: NoHand; 0: NoMove; 1: Move
            self.move_state = -1

            stateImg = cv2.putText(rgb.copy(), "No Hand", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                                    [255, 0, 0], 5)

        else:
            # Get the tracker
            self.state_parser.process([hTracker])
            stateImg = self.state_parser.plot_states(rgb.copy())

            # NOTE: The moving state is obtained here.
            # The return is supposed to be of the shape (N_state, ), where N_state is the number of states,
            # since it was designed to include extraction of all the states.
            # Since the puzzle states is implemented elsewhere, the N_state is 1, hence index [0]
            self.move_state = self.state_parser.get_states()[0]
            print('hand state',self.move_state)

        display_images_cv([stateImg[:, :, ::-1]], ratio=0.5, window_name="Move States")
    
    def PuzzleProgressTracking(self,call_back_id):
        if self.progress_window is None:
            self.progress_window = DynamicDisplay(
                    ParamDynamicDisplay(num=1, ylimit=100,                                        
                                        window_title='Puzzle Progress'))

        try:
            thePercent = self.puzzle_solver.progress(USE_MEASURED=False)
            self.progress_tracking[call_back_id]=thePercent
            #plot the percentage every x frames

            progress=sorted(self.progress_tracking.items())
            #print(progress)
            x,y=zip(*progress)
            y=list(y)
            x=list(x)
            if(len(y)>20):
                y=y[-20:]
                x=x[-20:]
            else:
                pad=np.zeros((20-len(y)))
                pad=list(pad)
                pad.extend(y)
                y=pad
                pad1=np.zeros((20-len(x)))
                pad1=list(pad1)
                pad1.extend(x)
                x=pad1
            #print('y',y)
            self.progress_window((call_back_id,y))
            if(self.ros_pub):
                return thePercent

        except:
            print('Double check the solution board to make it right.')    

    def PuzzleStateAnalysis(self, puzzle_solver_mode, postImg, visibleMask, hTracker_BEV, call_back_id):
        
        self.puzzle_solver_mode=puzzle_solver_mode
        
        if self.puzzle_solver_mode == 1:
            self.plan = self.puzzle_solver.calibrate(postImg, visibleMask, hTracker_BEV)

        elif (not(self.puzzle_solver_mode == 0 or self.puzzle_solver_mode == 2)):
            raise RuntimeError('Wrong puzzle_solver_mode!')
        
        else:    
            if self.puzzle_solver_mode == 0:

                if call_back_id == 0:
                    # Initialize the SolBoard using the very first frame.
                    self.puzzle_solver.setSolBoard(postImg)
                    print(f'Number of puzzle pieces registered in the solution board: {self.puzzle_solver.theManager.solution.size()}')
                # Plan not used yet
                self.plan = self.puzzle_solver.process(postImg, visibleMask, hTracker_BEV)

            elif self.puzzle_solver_mode == 2:
                print('call back id: ',call_back_id)
                
               
            
            # Initialize the SolBoard with saved board at the very first frame.
                if call_back_id == 0:
                    self.puzzle_solver.setSolBoard(postImg, self.params.puzzle_solver_SolBoard)

                    print(f'Number of puzzle pieces registered in the solution board: {self.puzzle_solver.theManager.solution.size()}')
                self.plan = self.puzzle_solver.process(postImg, visibleMask, hTracker_BEV, run_solver=True)
            
            if self.status_window is None and self.activity_window is None:
                    
                self.status_window = DynamicDisplay(
                    ParamDynamicDisplay(num=self.puzzle_solver.theManager.solution.size(),
                                        window_title='Status Change'))
                self.activity_window = DynamicDisplay(
                    ParamDynamicDisplay(num=self.puzzle_solver.theManager.solution.size(),
                                        status_label=['NONE', 'MOVE'], ylimit=1,
                                        window_title='Activity Change'))
                
            
        status_data = np.zeros(len(self.puzzle_solver.thePlanner.status_history))
        self.activity_data = np.zeros(len(self.puzzle_solver.thePlanner.status_history))

        for i in range(len(status_data)):
            try:
                status_data[i] = self.puzzle_solver.thePlanner.status_history[i][-1].value
            except:
                status_data[i] = PieceStatus.UNKNOWN.value

            if len(self.puzzle_solver.thePlanner.status_history[i]) >= 2 and \
                    self.puzzle_solver.thePlanner.status_history[i][-1] == PieceStatus.MEASURED and \
                    self.puzzle_solver.thePlanner.status_history[i][-2] != PieceStatus.MEASURED and \
                    np.linalg.norm(self.puzzle_solver.thePlanner.loc_history[i][-1] -
                                    self.puzzle_solver.thePlanner.loc_history[i][-2]) > 30:
                self.activity_data[i] = 1
                #print('Move activity detected.')

            else:
                self.activity_data[i] = 0

        self.status_window((call_back_id, status_data))
        self.activity_window((call_back_id, self.activity_data))

           
                



