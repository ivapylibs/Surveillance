"""
    @brief              The simulator for activity recognition testing

    @author             Yiye Chen.          yychen2019@gatech.edu
    @date               08/14/2021

"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Location():
    """
    The coordinate if from the lower left corner, to right and to up
    """
    x:float = -1
    y:float = -1

    def to_row_col(self, total_rows):
        """
        The (row, col) is from the upper left corner, to down and to right
        """
        row = total_rows - self.y
        col = self.x
        return row, col


class Puzzle_Piece():
    """
    The puzzle piece class. 
    Hardcode as a square
    """
    def __init__(self, size=15, color='g') -> None:
        # square size & centroid location
        self.location = Location()
        self.size = size
        self.assembled = False

    def draw(self, fh):
        pass


class Hand():
    """
    The class for hand controlling.
    Hardcode the hand as a circle
    """
    def __init__(self, radius=5, color='r'):

        # circle radius & centroid location
        self.location = Location()
        self.r = radius

        # color
        self.color = 'r'

        # a piece in hand?
        self.piece_in_hand = None
    
    def move(self):
        pass

    def pick(self, piece):
        pass

    def place(self, piece):
        pass

    def draw(self, fh):
        pass



class Simulator():
    """
    The simulator
    
    @param[in] N_piece          the number of puzzle pieces
    @param[in] size             The canvas size
    """
    def __init__(self, N_piece=1, size=100) -> None:

        self.size = size

        # initialize the piece position and the target piece position
        self.piece = [] 
        self.target_locations = []
        for i in range(N_piece):
            self.piece.append(Puzzle_Piece(size=15, color='g'))
            self.piece[i].location.x = int(3/4*self.size)
            self.piece[i].location.y = int((i+1) / (N_piece+1) * self.size)
            self.target_locations.append(Location(int(1/4*self.size), self.piece[i].location.y))
        
        # initialize the hand position
        self.hand = Hand()
        self.hand.location.x = int(self.size/2)
        self.hand.location.y = int(self.size/2)


    def simulate(self):
        pass

    def draw(self, fh):
        pass


if __name__ == "__main__":
    simulator = Simulator()
    simulator.simulate()