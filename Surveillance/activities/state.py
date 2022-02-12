"""

    @brief              The classes of taking the tracking/monitoring signals and converting to the human state,
                        which will be used to interpret activities

    @author             Yiye Chen.                  yychen2019@gatech.edu
    @date               08/12/2021

"""
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from Surveillance.activities.base import Base

class Base_state(Base):
    """
    The base class for the human state estimator.

    @param[in]  signal_number               how many signals will be used as input
    @param[in]  state_number                how many states will be estimated

    @param[in]  signal_cache_limit            How many previous signals to be stored. Default:1000
    @param[in]  state_cache_limit             How many previous states to be stored. Default:1000

    @param[in]  signal_names                A list of signal names. If empty then will assign signal_1, signal_2, ...
    @param[in]  state_names                 A list of state names. If empty then will assign state_1, state_2, ...
    """
    def __init__(self, signal_number, state_number, signal_cache_limit=1000, state_cache_limit=1000, 
                signal_names=[], state_names=[]):
        self.signal_number = signal_number                          
        self.state_number = state_number                            # how many states will be estimated

        self.signal_cache_limit = signal_cache_limit
        self.state_cache_limit = state_cache_limit
        self.signal_cache_count = 0                                 # count the number of cached signals
        self.state_cache_count = 0                                  # count the number of cached states

        self.state_names = state_names
        self.signal_names = signal_names

        # signal cache - 2d list of the shape (self.signal_number, self.signal_cache_limit)
        self.signals_cache = []
        for i in range(self.signal_number):
            self.signals_cache.append([])
        
        # state cache - since the current design are binary, make it array
        self.states_cache = np.empty((self.state_number, self.state_cache_limit))        # a list of the cached state list, each element of which represents a state 

        # store the figure handle index for display
        self.f_idx = None

    def measure(self, cur_signals:List):
        """
        The workflow of the signal2state 

        @param[in]  cur_signals.            An array of the signals, each element of which represents the new income of different signal types
        """
        assert len(cur_signals) == self.signal_number

        cur_states = self.parse(cur_signals)
        assert cur_states.size == self.state_number

        self.update(cur_signals, cur_states)

        self.signal_cache_count += 1
        self.state_cache_count += 1

    def parse(self, cur_signals):
        """
        Parse the state out of the signals. Return the states
        """
        raise NotImplementedError

    def update(self, cur_signals, cur_states):
        """
        Update the cache states with new ones
        """
        self._append_with_number_limit(self.signals_cache, cur_signals, self.signal_cache_limit, self.signal_cache_count)
        self._append_with_number_limit(self.states_cache, cur_states, self.state_cache_limit, self.state_cache_count)

    
    def visualize(self, fh=None):
        """
        Visualize the state process.
        Will only display the cached states, which is the latest state_cache_limit states

        @param[in] fh                   The figure handle. Default is None. When set to None then a new figure will be created
                                        Note that the fh will be stored at the first time being used, and all future drawing will be on that figure
        """
        # fetch the figure/ax
        if self.f_idx is None:
            if fh is None:
                fh = plt.figure()
            self.f_idx = fh.number
        else:
            fh = plt.figure(self.f_idx)

        # clear previous plot 
        plt.clf()

        # draw the new state
        for i in range(self.state_number):
            plt.subplot(self.state_number, 1, i+1) 
            plt.title("State - {}".format(self.state_names[i]))
            plt.plot(self.states_cache[i, :self.state_cache_count])
    
    def _append_with_number_limit(self, cache, new, num_limit, cur_count):
        """
        @param[in]  cur_count           The count BEFORE appendin the new
        """
        if isinstance(cache, np.ndarray):
            assert isinstance(new, np.ndarray) and cache.shape[0] == new.shape[0]
            if cur_count < num_limit:
                cache[:, cur_count] = new
            else:
                cache[:, :num_limit-1] = cache[:, 1:num_limit]
                cache[:, num_limit-1] = new
        elif isinstance(cache, list):
            assert isinstance(new, list) and len(cache) == len(new)
            if cur_count < num_limit:
                # iterate through the state/signal numbers
                for i in range(len(cache)):
                    cache[i].append(new[i])
            else:
                # iterate through the state/signal numbers
                for i in range(len(cache)):
                    # first pop out the first element then append new
                    cache[i].pop(0)
                    cache[i].append(new[i])

class StateEstimator(Base_state):
    """
    The State Estimator v1.0

    Fix to estimate three binary states: Move, Make_Puzzle_Progress, Puzzle_in_hand.
    The signal used for the state estimation is:
        1. The hand location,
        2. The puzzle solving progress (percentage)
        3. Piece in hand 
    For now allow customization for the signals and the states used for the development purpose
    """

    def __init__(self, signal_number, signal_cache_limit=1000, state_cache_limit=1000,signal_names=[],\
        state_number = 3, state_names=["Move", "Progress_Made", "Puzzle_in_Hand"],\
        move_th=2):
        super().__init__(signal_number, state_number=state_number, signal_cache_limit=signal_cache_limit, state_cache_limit=state_cache_limit,
                        signal_names=signal_names, state_names=state_names)
        
        self.move_th = 2

    def parse(self, cur_signals:List):
        """Parse the states out of the signals

        Args:
            cur_signals (np.ndarray, (signal_number, )):    The signals of the current timestamp. \
                According to the current design, the signals should be a list who stores: \
                    1. hand location (np.ndarray, (2, ))
                    2. Puzzle solving progress (percentage number)
                    3. puzzle piece in hand (binary)
        Return:
            cur_states (np.ndarray, (state_num, )):         The states parsed of the current timestamp
        """
        cur_states = np.zeros((self.state_number), dtype=bool)
        
        # parse move
        cur_states[0] = self.parse_move(cur_signals)

        return cur_states
    
    def parse_move(self, cur_signals):
        """Parse the moving state
        """

        # if the first signal, just return not moving
        if self.signal_cache_count == 0:
            return False

        # convention: The location is the first signal
        cur_location = cur_signals[0]
        last_location = self.signals_cache[0][-1]

        distance = np.linalg.norm(
            np.array(cur_location) - np.array(last_location)
        )

        return distance > self.move_th
    
    def parse_progress(self, cur_signals):
        """
        Parse the progress state, which is a binary indicator of whether the puzzle solving is making progress or not
        """
        raise NotImplementedError

    def parse_PinH(self, cur_signals):
        """
        Parse whether the puzzle-in-hand state, which is a binary indicator whether the hand is holding a puzzle piece or not
        """
        raise NotImplementedError

