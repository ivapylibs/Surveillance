"""

    @brief              The classes of taking the tracking/monitoring signals and converting to the human state,
                        which will be used to interpret activities

    @author             Yiye Chen.                  yychen2019@gatech.edu
    @date               08/12/2021

"""

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

        self.signals_cache = np.empty((self.signal_number, self.signal_cache_limit))       # a list of the cached signal list, each element of which represents a signal
        self.states_cache = np.empty((self.state_number, self.state_cache_limit))        # a list of the cached state list, each element of which represents a state 

        # store the figure handle index for display
        self.f_idx = None

    def measure(self, cur_signals):
        """
        The workflow of the signal2state 

        @param[in]  cur_signals.            An array of the signals, each element of which represents the new income of different signal types
        """
        if isinstance(cur_signals, list):
            cur_signals = np.array(cur_signals)
        assert cur_signals.size == self.signal_number

        cur_states = self.parse(cur_signals)
        assert cur_states.size == self.state_number

        self.update(cur_signals, cur_states)

        self.signal_cache_count += 1
        self.state_cache_count += 1

    def parse(self, cur_signals):
        """
        Parse the state out of the signals
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
        if cur_count < num_limit:
            cache[:, cur_count] = new
        else:
            cache[:, :num_limit-1] = cache[:, 1:num_limit]
            cache[:, num_limit-1] = new

class StateEstimator(Base_state):
    """
    The State Estimator v1.0

    Fix to estimate three binary states: Move, Make_Puzzle_Progress, Puzzle_in_hand
    For now allow customization for the signals to be used.
    """

    def __init__(self, signal_number, signal_cache_limit=1000, state_cache_limit=1000,signal_names=[]):
        super().__init__(signal_number, state_number=3, signal_cache_limit=state_cache_limit, state_cache_limit=state_cache_limit,
                        signal_names=signal_names, state_names=["Move", "Progress_Made", "Puzzle_in_Hand"])

    def parse(self, signals):
        pass