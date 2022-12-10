# ========================= Surveillance.activity.piece_status_change ========================
#
# @class    Surveillance.activity.piece_status_change
#
# @brief    This is the piece status change idea to analyze the piece move activity
#
# ========================= Surveillance.activity.piece_status_change ========================
#
# @file     piece_status_change.py
#
# @author   Yunzhi Lin,             yunzhi.lin@gatech.edu
# @date     2022/12/10 [created]
#
#
# ========================= Surveillance.activity.piece_status_change ========================

import numpy as np

from puzzle.piece.template import PieceStatus

from Surveillance.deployment.utils import find_last_occurance

def piece_status_change(status_history, loc_history, activity_history):
    """
    This function is to analyze the piece status change

    Args:
        status_history: The status history of the piece.
        loc_history: The location history of the piece.
        activity_history: The activity history of the piece.

    Returns:
        status_data: The status data of the piece.
        activity_data : The activity data of the piece.
    """

    # status/activity data is for the current frame
    status_data = np.zeros(len(status_history))
    activity_data = np.zeros(len(status_history))

    for i in range(len(status_data)):
        try:
            status_data[i] = status_history[i][-1].value
        except:
            status_data[i] = PieceStatus.UNKNOWN.value

        moveDetect = False
        if len(status_history[i]) >= 2:

            # 1) Transition from not MEASURED to MEASURED and location changes
            if (status_history[i][-1] == PieceStatus.MEASURED and \
                    status_history[i][-2] != PieceStatus.MEASURED and \
                    np.linalg.norm(loc_history[i][-1] - loc_history[i][-2]) > 30):

                # move_dis = np.linalg.norm(loc_history[i][-1] -
                #                loc_history[i][-2])
                # print(f"Move dis: {move_dis}")
                moveDetect = True
                activity_data[i] = 1
                activity_history[i].append(1)
                print(f'Move activity detected for piece {i}')

            # 2) Transition from GONE to MEASURED (there might be some frame gap, e.g., GONE-> INVISIBLE -> INVISIBLE -> MEASURED)
            elif status_history[i][-1] == PieceStatus.MEASURED:

                # Find the index of the last GONE status of the piece from the status history
                index = find_last_occurance(status_history[i], PieceStatus.GONE)
                if index is not None:

                    # Check if there is not any move activity in the activity history
                    if 1 not in activity_history[i][index:]:
                        moveDetect = True
                        activity_data[i] = 1
                        activity_history[i].append(1)
                        print(f'Move activity detected for piece {i}')

        if moveDetect is False:
            activity_data[i] = 0
            activity_history[i].append(0)

    return status_data, activity_data