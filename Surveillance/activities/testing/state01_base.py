"""

    @ brief         Test the functionality of the base State class with simple state extraction policy and 
                    synthetic signals

    @author         Yiye Chen.          yychen2019@gatech.edu
    @date           08/13/2021

"""


from Surveillance.activities.state import Base

# === [1] Create a simple state parser 
class State_simple(Base):
    """
    Create a simple state estimator
    """
    def __init__(self, state_names=["Same", "Reverse"]):
        super().__init__(state_names=state_names)
    
    def parse(self, cur_signals):
        raise NotImplementedError()

state_parser = State_simple()

# === [2] create synthetic signals

# === [3] Parse and visualize
