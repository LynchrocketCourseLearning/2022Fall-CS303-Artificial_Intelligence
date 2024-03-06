import random

import numpy as np

from Project1.agent import AI
from Project1.agent_ab import AI as AI_ab


def check(chessboard):
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == 0:
                return True
    return False


def diff(chessboard):
    black = 0
    white = 0
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == 1:
                white = white + 1
            else:
                black = black + 1
    return white - black


def take_step(chessboard_o, player, step):
    opponents = []
    # print(chessboard_o)
    chessboard = np.copy(chessboard_o)
    chessboard[step[0]][step[1]] = player
    # up
    for i in range(1, 8):
        if step[0] - i >= 0:
            if chessboard[step[0] - i][step[1]] == - player:
                opponents.append((step[0] - i, step[1]))
            elif chessboard[step[0] - i][step[1]] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break
    # down
    for i in range(1, 8):
        if step[0] + i <= 7:
            if chessboard[step[0] + i][step[1]] == - player:
                opponents.append((step[0] + i, step[1]))
            elif chessboard[step[0] + i][step[1]] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break
    # left
    for i in range(1, 8):
        if step[1] - i >= 0:
            if chessboard[step[0]][step[1] - i] == - player:
                opponents.append((step[0], step[1] - i))
            elif chessboard[step[0]][step[1] - i] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break
    # right
    for i in range(1, 8):
        if step[1] + i <= 7:
            if chessboard[step[0]][step[1] + i] == - player:
                opponents.append((step[0], step[1] + i))
            elif chessboard[step[0]][step[1] + i] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break
    # up right
    for i in range(1, 8):
        if step[0] - i >= 0 and step[1] + i <= 7:
            if chessboard[step[0] - i][step[1] + i] == - player:
                opponents.append((step[0] - i, step[1] + i))
            elif chessboard[step[0] - i][step[1] + i] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break
    # up left
    for i in range(1, 8):
        if step[0] - i >= 0 and step[1] - i >= 0:
            if chessboard[step[0] - i][step[1] - i] == - player:
                opponents.append((step[0] - i, step[1] - i))
            elif chessboard[step[0] - i][step[1] - i] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break
    # down right
    for i in range(1, 8):
        if step[0] + i <= 7 and step[1] + i <= 7:
            if chessboard[step[0] + i][step[1] + i] == - player:
                opponents.append((step[0] + i, step[1] + i))
            elif chessboard[step[0] + i][step[1] + i] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break
    # down left
    for i in range(1, 8):
        if step[0] + i <= 7 and step[1] - i >= 0:
            if chessboard[step[0] + i][step[1] - i] == - player:
                opponents.append((step[0] + i, step[1] - i))
            elif chessboard[step[0] + i][step[1] - i] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break

    # print(chessboard_o)
    return chessboard


if __name__ == '__main__':

    count = 0
    current_player = -1
    time_out = 3

    # initialize the chessboard
    chessboard = np.zeros((8, 8))
    chessboard[3][3] = 1
    chessboard[4][4] = 1
    chessboard[3][4] = -1
    chessboard[4][3] = -1

    while check(chessboard):
        ai = AI(8, current_player, 3)
        ai.go(chessboard)
        steps = ai.candidate_list
        if len(steps) != 0:  # 如果有地方下
            take_step(chessboard, current_player, steps[len(steps) - 1])

        current_player = - current_player  # change player

        weaker = AI_ab(8, current_player, 3)
        weaker.go(chessboard)
        wsteps = weaker.candidate_list
        if len(wsteps) != 0:
            step = wsteps[(len(wsteps) - 1)]
            take_step(chessboard, current_player, step)
        # print(step)
        current_player = - current_player
        count = count + 1
    print(chessboard)
    if diff(chessboard) > 0:
        print("Black Win : AI")
    else:
        print("White Win: AI_ab")
