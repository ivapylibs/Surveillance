"""

    @brief              The classes of taking the tracking/monitoring signals and converting to the human state,
                        which will be used to interpret activities

    @author             Yiye Chen.                  yychen2019@gatech.edu
    @date               08/12/2021

"""

import matplotlib.pyplot as plt
import numpy as np

class Base(object):
    """
    The base class for the human state estimator.

    @param[in]  signal_number               how many signals will be used as input
    @param[in]  state_number                how many states will be estimated

    @param[in]  signal_cache_num            How many previous signals to be stored. Default:1000
    @param[in]  state_cache_num             How many previous states to be stored. Default:1000

    @param[in]  signal_names                A list of signal names. If empty then will assign signal_1, signal_2, ...
    @param[in]  state_names                 A list of state names. If empty then will assign state_1, state_2, ...
    """
    def __init__(self, signal_number, state_number, signal_cache_num=1000, state_cache_num=1000, 
                signal_names=[], state_names=[]):
        self.signal_number = signal_number                          
        self.state_number = state_number                            # how many states will be estimated
        self.signals_cache = np.empty((self.signal_number, ))    # a list of the cached signal list, each element of which represents a signal
        self.states_cache = np.empty((self.state_number, 0))     # a list of the cached state list, each element of which represents a state 

        self.state_names = self.state_names
        self.signal_names = self.signal_names

    def measure(self, signals):
        """
        The workflow of the signal2state 

        @param[in]  signals.            A list/array of the signals, each element of which represents the new income of different signal types
        """
        cur_states = self.parse(signals)
        self.update(cur_states)

    def parse(self, signals):
        """
        Parse the state out of the signals
        """
        raise NotImplementedError

    def update(self, cur_states):
        """
        Update the cache states with new ones
        """
        raise NotImplementedError
    
    def visualize_rt(self, time_delay=0.03, axes=None):
        """
        Visualize the state process in realtime.
        It will visualize the new state.

        As a real-time visualizer, it should be called in the following way:

        while(receiving_signals)
            state_estimator.measure(new_signal)
            state_estimator.visualize_rt(time_delay=0.)

        @param[in] time_delay           The delay of time for visualization
        @param[in] axes                 The list of axes to visualize each 
        """
        plt.pause(time_delay)

        # fetch the figure/ax

        # draw the new state

        raise NotImplementedError
    
    def _append_with_number_limit(self, cache, new):
        """
        """
        pass