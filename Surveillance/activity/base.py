"""
    @ brief                 The base class for the state_parser and the the activity parser

    @ author                Yiye Chen.          yychen2019@gatech.edu
    @ date                  08/19/2021
"""

class Base(object):

    def __init__(self):
        self.x = None
        pass

    def process(self, signal):
        """
        Process the new income signal
        """
        self.predict()
        self.measure(signal)
        self.correct()
        self.adapt()

    def predict(self):
        return None

    def measure(self, signal):
        return None

    def correct(self):
        return None
    
    def adapt(self):
        return None
