"""

    @ brief         Test the functionality of the base State class with simple state extraction policy and 
                    synthetic signals

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           08/13/2021

"""

import numpy as np
import matplotlib.pyplot as plt

from Surveillance.activity.state import Base_state

# === [1] Create a simple state parser 
class State_simple(Base_state):
    """
    Create a simple state estimator
    """
    def __init__(self, signal_cache_limit, state_cache_limit, state_names=["Same", "Reverse"]):
        super().__init__(signal_number=1, state_number=2, signal_cache_limit=signal_cache_limit, state_cache_limit=state_cache_limit,
                        state_names=state_names)
    
    def parse(self, cur_signals):
        cur_state = np.empty((2,))
        cur_state[0] = cur_signals[0]
        cur_state[1] = np.logical_not(cur_signals[0])
        return cur_state

state_parser = State_simple(signal_cache_limit=1000, state_cache_limit=20)

# === [2] create synthetic signals
N = 100
signals = np.ones(N, dtype=bool)
index = np.random.choice(signals.shape[0], N//2, replace=False)
signals[index] = False

# === [3] Parse and visualize
plt.figure(1)
plt.title("The signal")
plt.plot(signals)
for i in range(N):
    signal = np.array(signals[i]).reshape(1)
    # plot a vertical line to indicate where we are
    plt.figure(1)
    line = plt.axvline(x=i, color='r')

    # parse state
    state_parser.process([signal])
    state_parser.visualize()

    # draw
    plt.draw()
    plt.pause(0.5)
    line.remove()

