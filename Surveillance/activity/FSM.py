# ========================= Surveillance.activity.FSM ========================
#
# @class    Surveillance.activity.FSM
#
# @brief    This is the basic finite state machine implementation.
#           Note:
#           Right now, it does not help that much.
#           Only useful when the user's behavior follows the designed pattern.
#           An alternative way is to directly rely on the puzzle_solver.
#           However, it is hard to tune the move_th.
#
# ========================= Surveillance.activity.FSM ========================
#
# @file     FSM.py
#
# @author   Yunzhi Lin,             yunzhi.lin@gatech.edu
# @date     2022/03/22 [created]
#
#
# ========================= Surveillance.activity.FSM ========================

from transitions import Machine

class Pick(Machine):
    def __init__(self):
        # The states argument defines the name of states
        states = ['A', 'B', 'C', 'D', 'E']

        # The trigger argument defines the name of the new triggering method
        transitions = [
            {'trigger': 'stop', 'source': 'A', 'dest': 'A'},
            {'trigger': 'move', 'source': 'A', 'dest': 'B'},
            {'trigger': 'move', 'source': 'B', 'dest': 'B'},
            {'trigger': 'stop', 'source': 'B', 'dest': 'C'},
            {'trigger': 'move', 'source': 'C', 'dest': 'D'},
            {'trigger': 'stop', 'source': 'C', 'dest': 'C'},
            {'trigger': 'no_piece_disappear', 'source': 'D', 'dest': 'B'},
            {'trigger': 'piece_disappear', 'source': 'D', 'dest': 'E'},
            {'trigger': 'reset', 'source': '*', 'dest': 'A'}
        ]

        Machine.__init__(self, states=states, transitions=transitions, initial='A')

# Almost similar to Pick
class Place(Machine):
    def __init__(self):
        # The states argument defines the name of states
        states = ['A', 'B', 'C', 'D', 'E']

        # The trigger argument defines the name of the new triggering method
        transitions = [
            {'trigger': 'stop', 'source': 'A', 'dest': 'A'},
            {'trigger': 'move', 'source': 'A', 'dest': 'B'},
            {'trigger': 'move', 'source': 'B', 'dest': 'B'},
            {'trigger': 'stop', 'source': 'B', 'dest': 'C'},
            {'trigger': 'move', 'source': 'C', 'dest': 'D'},
            {'trigger': 'stop', 'source': 'C', 'dest': 'C'},
            {'trigger': 'no_piece_added', 'source': 'D', 'dest': 'B'},
            {'trigger': 'piece_added', 'source': 'D', 'dest': 'E'},
            {'trigger': 'reset', 'source': '*', 'dest': 'A'}
        ]

        Machine.__init__(self, states=states, transitions=transitions, initial='A')

if __name__ == "__main__":

    pick_model = Pick()

    # Test
    print(pick_model.state)    # A
    pick_model.stop()
    pick_model.stop()
    pick_model.move()
    pick_model.move()
    pick_model.stop()
    print(pick_model.state)

# ========================= Surveillance.activity.FSM ========================
