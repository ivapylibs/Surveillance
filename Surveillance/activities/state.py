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
    """

    def __init__(self, state_names=None):
        self.signals_hist = []      # a list of the cached signal list, each element of which represents a signal
        self.states_hist = []       # a list of the cached state list, each element of which represents a state 

    def measure(self, signals:list):
        """
        The workflow of the signal2state 

        @param[in]  signals.            A list of the signals
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
    
    def visualize_rt(self, time_delay=0.5):
        """
        Visualize the state process in realtime.

        @param[in] time_delay           The delay of time for visualization
        """
        plt.pause(time_delay)
        # draw the new state
        raise NotImplementedError