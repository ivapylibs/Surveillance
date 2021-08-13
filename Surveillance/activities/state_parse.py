"""

    @brief              A collection of state parsing methods

    @author             Yiye Chen.              yychen2019@gatech.edu
    @date               08/13/2021

"""


def parse_move(prev_centroid, cur_centroid, threshold):
    """
    Parse the movement state, which is a binary indicator of whether the hand is moving or not
    """
    raise NotImplementedError


def parse_progress(prev_prog, cur_prog):
    """
    Parse the progress state, which is a binary indicator of whether the puzzle solving is making progress or not

    @param[in]  prev_prog               Previous percentage of how many puzzle has been assembled
    @param[in]  cur_prog                Current percentage of how many puzzle has been assembled
    """
    return  cur_prog > prev_prog

def parse_PinH():
    """
    Parse whether the puzzle-in-hand state, which is a binary indicator whether the hand is holding a puzzle piece or not
    """
    raise NotImplementedError