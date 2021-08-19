"""
    @brief              The simulator for activity recognition testing

    @author             Yiye Chen.          yychen2019@gatech.edu
    @date               08/14/2021

"""
from Surveillance.activities.state_parse import parse_progress
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math

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
    def __init__(self, size=15, color='b') -> None:
        # square size & centroid location & appearance
        self.location = Location()
        self.size = size
        self.color = color

        # state
        self.assembled = False
        self.picked = False

    def draw(self, ax):
        # the xy parameter is the left bottom corner
        rect = Rectangle(
            xy=[self.location.x - self.size/2, self.location.y - self.size/2],
            width=self.size,
            height=self.size,
            color=self.color
        )
        ax.add_patch(rect)

class Hand():
    """
    The class for hand controlling.
    Hardcode the hand as a circle
    """
    def __init__(self, radius=5, color='r', speed=5, pick_place_time=3):

        # circle radius & centroid location
        self.location = Location()
        self.r = radius

        # color
        self.color = color

        # a piece in hand?
        self.pieces_in_hand = None

        # speed
        self.speed = speed

        # how long will the hand stop when picking and placing
        self.stop_timer = -1
        self.pick_place_time = pick_place_time
    
    def move(self, target:Location, delta_t, piece=None):
        """
        If cannot reach the target within the time interval, then move to the furthest
        If can, then stop at the target
        """
        # distance
        delta_x = target.x - self.location.x
        delta_y = target.y - self.location.y
        distance = math.sqrt(delta_x**2 + delta_y**2)

        if distance >= delta_t * self.speed:
            x_step = delta_x * (delta_t * self.speed) / distance
            y_step = delta_y * (delta_t * self.speed) / distance
            self.location.x += x_step
            self.location.y += y_step
        else:
            self.location.x = target.x
            self.location.y = target.y
        
        # move the piece together
        if piece is not None:
            piece.location.x = self.location.x
            piece.location.y = self.location.y

    def pick(self, piece, delta_t):
        # if haven't started to pick
        if self.stop_timer == -1:
            self.stop_timer = self.pick_place_time
            self.stop_timer -= delta_t
        # have started to pick. Just stop for enough time and pick up
        elif self.stop_timer > 0:
            self.stop_timer -= delta_t
            if self.stop_timer <= 0:
                piece.picked = True
                self.stop_timer = -1
        else:
            raise RuntimeError("Shouldn't have reached this state")

    def place(self, piece, delta_t):
        """
        For now assuming all place results in assembly
        TODO: add place without assembly
        """
        # if haven't started to place
        if self.stop_timer == -1:
            self.stop_timer = self.pick_place_time
            self.stop_timer -= delta_t
        # have started to place. Just stop for enough time and place
        elif self.stop_timer > 0:
            self.stop_timer -= delta_t
            if self.stop_timer <= 0:
                piece.assembled = True
                piece.placed = False
                self.stop_timer = -1
        else:
            raise RuntimeError("Shouldn't have reached this state")

    def draw(self, ax):
        circle = plt.Circle((self.location.x, self.location.y), self.r, color=self.color)
        ax.add_patch(circle)



class Simulator():
    """
    The simulator
    
    @param[in] N_piece          the number of puzzle pieces
    @param[in] size             The canvas size
    """
    def __init__(self, N_piece=1, size=100, speed=5, hand_stop_time=3):

        self.size = size
        self.speed = speed

        # initialize the piece position and the target piece position
        self.pieces = [] 
        self.target_locations = []
        for i in range(N_piece):
            self.pieces.append(Puzzle_Piece(size=15))
            self.pieces[i].location.x = int(3/4*self.size)
            self.pieces[i].location.y = int((i+1) / (N_piece+1) * self.size)
            self.target_locations.append(Location(int(1/4*self.size), self.pieces[i].location.y))

        # the observable pieces
        self.pieces_on_table = self._get_pieces_on_table()
        
        # initialize the hand position
        self.hand = Hand(speed = speed, pick_place_time=hand_stop_time)
        self.hand.location.x = int(self.size/2)
        self.hand.location.y = int(self.size/2)


    def simulate_step(self, delta_t):
        """
        @param[in]  delta_t         The time length for one simulation step
        @param[out] finish          Binary. Finished or not
        """
        # find the next piece to deal with
        finish = True
        for piece in self.pieces:
            if not piece.assembled:
                finish = False # if one of the pieces is not assembled, then haven't finished yet.
                break
        
        if not piece.picked:
            if piece.location != self.hand.location:
                # reach to a piece to pick
                self.hand.move(piece.location, delta_t)
            elif piece.location == self.hand.location:
                # pick up a piece
                self.hand.pick(piece, delta_t)
        elif piece.picked:
            target_location = Location(x=1/4*self.size, y=piece.location.y)
            if piece.location != target_location:
                # go to the target position
                self.hand.move(target_location, delta_t, piece=piece)
            elif piece.location == target_location:
                # place the piece
                #For now assuming all place result in assembly
                #TODO: add place without assembly
                self.hand.place(piece, delta_t)
        return finish

    def draw(self, ax):
        # draw the target line
        ax.axvline(x=1/4*self.size, linestyle='--', color='g')        

        # draw pieces
        for i in range(len(self.pieces)):
            self.pieces[i].draw(ax)
        
        # draw the hand
        self.hand.draw(ax)

    
    def _get_pieces_on_table(self):
        return [p for p in self.pieces if not p.picked]

if __name__ == "__main__":
    # simulator
    simulator = Simulator(N_piece=4, speed=200, hand_stop_time=0.5)

    # simulation variables
    finish_flag = False
    delta_t = 0.01

    # figure for the visualization
    fh, ax = plt.subplots()
    ax.set_xlim([0, simulator.size])
    ax.set_ylim([0, simulator.size])

    while(not finish_flag):
        # simulate a step
        finish_flag = simulator.simulate_step(delta_t)

        # remove previous draw
        for artist in ax.patches:
            artist.remove()
        # draw the current step
        simulator.draw(ax)
        plt.draw()
        plt.pause(delta_t)