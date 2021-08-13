"""

    @brief              The classes of taking the tracking/monitoring signals and converting to the human state,
                        which will be used to interpret activities

    @author             Yiye Chen.                  yychen2019@gatech.edu
    @date               08/12/2021

"""

class Base(object):
    """
    The base class for the human state estimator.
    """

    def __init__(self):
        self.signals_hist = []
        self.states_hist = []

    def measure(self, signals:list):
        """
        The workflow of the signal2state 

        @param[in]  signals.            A list of the signals
        """
        cur_states = self.parse(signals)
        self.update(cur_states)

    def parse(self):
        """
        Parse the state out of the signals
        """
        raise NotImplementedError

    def update(self, cur_states):
        """
        Update the cache states with new ones
        """
        raise NotImplementedError
    
    def visualize(self):
        """
        Visualize the state process
        """
        raise NotImplementedError