"""

    @brief          The ROS Wrapper for the Surveillance.Monitor class

    @author         Nimisha Pabbichetty,          npabbichetty3@gatech.edu
    @date           

"""
import rospy
import yaml
import cv2
import copy
import numpy as np

from dataclasses import dataclass
from benedict import benedict

from std_msgs.msg import String, Float64

from ROSWrapper.subscribers.Images_sub import Images_sub
from ROSWrapper.subscribers.String_sub import String_sub
from ROSWrapper.publishers.Image_pub import Image_pub

from surveillance.msg import handState, activityUpdate

from Surveillance.deployment.Monitor import SurveillanceMonitor
from Surveillance.activity.state import StateEstimator
from Surveillance.activity.utils import DynamicDisplay, ParamDynamicDisplay

from puzzle.runner import RealSolver
from puzzle.utils.dataProcessing import convert_ROS2dict, convert_dict2ROS

from camera.utils.display import display_images_cv

@dataclass
class Params:
    #TODO: put these in the YAML file
    #list of topics -> use in ROS Wrapper instead, still include in YAML tho, create 2 YAMLs and merge in wrapper, one basic for monitor and one with ROS stuff
    test_rgb_topic: str = "/test_rgb"
    test_dep_topic: str = "/test_depth"
    test_activity_topic: str = "/test_activity"

    # Publish
    postImg_topic: str = "postImg"
    visibleMask_topic: str = "visibleMask"
    hTracker_BEV_topic: str = "hTracker_BEV"

    # Subscribe, the name needs to be consistent with the one in the puzzle solver part
    # See https://github.com/ivapylibs/puzzle_solver/tree/yunzhi/puzzle/testing/real_runnerROS.py
    puzzle_solver_info_topic: str = "/puzzle_solver_info"
    status_history_topic: str = "/status_history"
    loc_history_topic: str = "/loc_history"

    # @note Not that important in this module, just for display, maybe add later
    bMeasImage_topic: str = "/bMeasImage"
    bTrackImage_topic: str = "/bTrackImage"
    bTrackImage_SolID_topic: str = "/bTrackImage_SolID"

    hand_state_topic: str = "/Monitor/HandState"
    puzzle_tracking_topic: str = "/Monitor/PuzzleTracking"
    activity_update_topic: str = "/Monitor/ActivityUpdate"

    fDir: str = "./"
    rosbag_name: str = "testing/data/tangled_1_work.bag"
    real_time: bool = False
    display: str = '001000'
    puzzle_solver_SolBoard: str = "testing/data/caliSolBoard.obj"

    reCalibrate: bool=False


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

class MonitorRunner(SurveillanceMonitor):
    def __init__(self, puzzle_solver: RealSolver(), state_parser: StateEstimator(signal_number=1), params: Params = Params()):
        
        self.mparams=params
        self.puzzle_solver=puzzle_solver
        self.state_parser=state_parser
        super().__init__(puzzle_solver = self.puzzle_solver, state_parser = self.state_parser)
        #self.monitor = SurveillanceMonitor(puzzle_solver = self.puzzle_solver, state_parser = self.state_parser, params = self.params)
        
        self.RGB_np = None
        self.Mask_np = None
        self.hTracker_BEV = None

        self.rgb_frame_stamp = None
        self.rgb_frame_stamp_prev = None
        self.ros_pub = True #put this into YAML

        self.thePercent=0.0

        #move this block to the puzzle state analysis block or the higher level wrapper, TBD
        
        self.hand_state_pub = rospy.Publisher(self.mparams.hand_state_topic, handState, queue_size=1)
        self.puzzle_tracking_pub = rospy.Publisher(self.mparams.puzzle_tracking_topic, Float64, queue_size=1)
        self.activity_update_pub = rospy.Publisher(self.mparams.activity_update_topic, activityUpdate, queue_size=1)
        
        self.puzzle_solver_info_pub = rospy.Publisher(self.mparams.puzzle_solver_info_topic, String, queue_size=5)
        self.status_history_pub = rospy.Publisher(self.mparams.status_history_topic, String, queue_size=5)
        self.loc_history_pub = rospy.Publisher(self.mparams.loc_history_topic, String, queue_size=5)

        self.bMeasImage_pub = Image_pub(topic_name=self.mparams.bMeasImage_topic)
        self.bTrackImage_pub = Image_pub(topic_name=self.mparams.bTrackImage_topic)
        self.bTrackImage_SolID_pub = Image_pub(topic_name=self.mparams.bTrackImage_SolID_topic)

        self.init_solution_flag=False
        Images_sub([self.mparams.postImg_topic, self.mparams.visibleMask_topic], callback_np=self.callback_rgbMask)

        # Initialize a subscriber for other info
        String_sub(self.mparams.hTracker_BEV_topic, String, callback_np=self.callback_hTracker_BEV)

    def callback_rgbMask(self, arg_list):
        RGB_np = arg_list[0]
        Mask_np = arg_list[1]
        rgb_frame_stamp = arg_list[2].to_sec()

        self.RGB_np = RGB_np.copy()
        self.Mask_np = Mask_np.copy()
        self.rgb_frame_stamp = copy.deepcopy(rgb_frame_stamp)

    def callback_hTracker_BEV(self, msg):

        info_dict = convert_ROS2dict(msg)
        hTracker_BEV = info_dict['hTracker_BEV']

        self.hTracker_BEV = hTracker_BEV if hTracker_BEV is None else np.array(hTracker_BEV)

    def HandStateAnalysis(self,hTracker,rgb):
        
        hand_state_msg=handState()
        if hTracker is None:
                    # -1: NoHand; 0: NoMove; 1: Move
            self.move_state = -1
            hand_state_msg.state_name = "No Hand" 

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

            if(self.move_state==0):
                hand_state_msg.state_name = "No Move"
            else:
                hand_state_msg.state_name = "Move"  

        display_images_cv([stateImg[:, :, ::-1]], ratio=0.5, window_name="Move States")
        hand_state_msg.hand_state=self.move_state

        self.hand_state_pub.publish(hand_state_msg)
    
    def runPuzzleProgressTracking(self, call_back_id):
        #the puzzle progress is not accurate, need to discuss with Yunzhi as to why
        self.thePercent = self.PuzzleProgressTracking(call_back_id)
        self.puzzle_tracking_pub.publish(self.thePercent)
    
    def publishToROS(self):
        plan_processed = []
        for command in self.plan:
            if command:
                plan_processed.append([command[0], command[1], command[2], command[3].tolist()])
            else:
                plan_processed.append(command)

        # status_history  e.g., k: [PieceStatus(Enum), PieceStatus(Enum), ...]
        status_history_processed = {}
        for k, v in self.puzzle_solver.thePlanner.status_history.items():
            status_history_processed[k] = [x.value for x in v]

        # loc_history  e.g., k: [array([x1, y1]), array([x2, y2]), ...]
        loc_history_processed = {}
        for k, v in self.puzzle_solver.thePlanner.loc_history.items():
            loc_history_processed[k] = [x.tolist() for x in v]


        # Wrap the board into a dictionary
        info_dict ={
            'plan': plan_processed,
            'solution_board_size': self.puzzle_solver.theManager.solution.size(),
            'progress': self.thePercent,
        }

        # Publish the messages
        self.puzzle_solver_info_pub.publish(convert_dict2ROS(info_dict))
        self.status_history_pub.publish(convert_dict2ROS(status_history_processed))
        self.loc_history_pub.publish(convert_dict2ROS(loc_history_processed))

        # Publish the board images
        self.bMeasImage_pub.pub(self.puzzle_solver.bMeasImage)
        self.bTrackImage_pub.pub(self.puzzle_solver.bTrackImage)
        self.bTrackImage_SolID_pub.pub(self.puzzle_solver.bTrackImage_SolID)


    def runPuzzleStateAnalysis(self,call_back_id):
        
        if call_back_id==0:
            self.init_solution_flag=True
        if self.RGB_np is None:
            print("array empty, ",call_back_id)
            return
        rgb_frame_stamp = copy.deepcopy(self.rgb_frame_stamp)

            # Skip images with the same timestamp as the previous one
        if rgb_frame_stamp != None and self.rgb_frame_stamp_prev == rgb_frame_stamp:

            time.sleep(0.001)
            # if self.opt.verbose:
            #     print('Same timestamp')
            return
        else:
            print("array not empty, ",call_back_id)
            self.rgb_frame_stamp_prev = rgb_frame_stamp
            RGB_np = self.RGB_np.copy()
            Mask_np = self.Mask_np.copy()
            hTracker_BEV = copy.deepcopy(self.hTracker_BEV)

            if self.mparams.reCalibrate:
                self.PuzzleStateAnalysis(puzzle_solver_mode=1, postImg=RGB_np, visibleMask=Mask_np, hTracker_BEV=hTracker_BEV, call_back_id=call_back_id)
            else:
                if  self.init_solution_flag:
                    print('Initializing the solution board...')
                    self.puzzle_solver.setSolBoard(RGB_np, self.params.puzzle_solver_SolBoard)
                    self.init_solution_flag = False
                self.PuzzleStateAnalysis(puzzle_solver_mode=2, postImg=RGB_np, visibleMask=Mask_np, hTracker_BEV=hTracker_BEV, call_back_id=call_back_id)
                if self.plan is not None:
                    self.publishToROS()

            for i in range(len(self.activity_data)):
                if self.activity_data[i]==1:
                    update_msg=activityUpdate()
                    update_msg.piece_number=i
                    update_msg.activity='Move activity detected with piece '+str(update_msg.piece_number)
                    self.activity_update_pub.publish(update_msg)

   
        

