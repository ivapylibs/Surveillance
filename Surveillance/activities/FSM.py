# ========================= Surveillance.activities.FSM ========================
#
# @class    Surveillance.activities.FSM
#
# @brief    This is the simplest finite state machine implementation.
#
# ========================= Surveillance.activities.FSM ========================
#
# @file     FSM.py
#
# @author   Yunzhi Lin,             yunzhi.lin@gatech.edu
# @date     2022/03/22 [created]
#
#
# ========================= Surveillance.activities.FSM ========================

from transitions import Machine

class Pick(object):
    pass

model = Pick()

#The states argument defines the name of states
states=['A', 'B', 'C', 'D', 'E']

# The trigger argument defines the name of the new triggering method
transitions = [
    {'trigger': 'stop', 'source': 'A', 'dest': 'A' },
    {'trigger': 'move', 'source': 'A', 'dest': 'B'},
    {'trigger': 'move', 'source': 'B', 'dest': 'B'},
    {'trigger': 'stop', 'source': 'B', 'dest': 'C'},
    {'trigger': 'move', 'source': 'C', 'dest': 'D'},
    {'trigger': 'move', 'source': 'C', 'dest': 'D'},
    {'trigger': 'no_piece_disappear', 'source': 'D', 'dest': 'B'},
    {'trigger': 'piece_disappear', 'source': 'D', 'dest': 'E'}
]

machine = Machine(model=model, states=states, transitions=transitions, initial='A')

# Test
print(model.state)    # A
model.stop()
model.stop()
model.move()
model.move()
model.stop()
print(model.state)

# ========================= Surveillance.activities.FSM ========================
