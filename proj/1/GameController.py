import random

import numpy as np
# from agent_ab import AI
from . import AI2

def check(chessboard):
    for i in range(8):
        for j in range(8):
            if chessboard[i][j] == 0:
                return True
    return False

# >0 -->black win
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


def take_step(chessboard, player, step):
    opponents = []
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
                opponents.append((step[0] - i, step[1]))
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
                opponents.append((step[0] - i, step[1]))
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
                opponents.append((step[0] - i, step[1]))
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
                opponents.append((step[0] - i, step[1]))
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
                opponents.append((step[0] - i, step[1]))
            elif chessboard[step[0] + i][step[1] - i] == player:
                for e in opponents:
                    chessboard[e[0]][e[1]] = player
                opponents.clear()
                break
            else:
                opponents.clear()
                break


if __name__ == '__main__':
    g = np.random.randint(0, 28, (10, 35))
    genes = g.tolist()

    # initial chessboard
    # chessboard = np.zeros((8, 8))
    # chessboard[3][3] = 1
    # chessboard[4][4] = 1
    # chessboard[3][4] = -1
    # chessboard[4][3] = -1

    count = 0
    current_player = -1
    time_out = 3



    for iteration in range(1, 10):
        print("Generation :", iteration)
        evaluations = []
        for i in range(len(genes)):

            chessboard = np.zeros((8, 8))
            chessboard[3][3] = 1
            chessboard[4][4] = 1
            chessboard[3][4] = -1
            chessboard[4][3] = -1
            count = 0
            while check(chessboard):
                ai = AI2.AI(8, current_player, 3)
                ai.go(chessboard)
                steps = ai.candidate_list
                if len(steps) != 0:
                    take_step(chessboard, current_player, steps[len(steps) - 1])

                current_player = - current_player
                weaker = AI2.AI(8, current_player, 3)
                weaker.go(chessboard)
                wsteps = weaker.candidate_list
                if len(wsteps) != 0:
                    step = wsteps[genes[i][count] % (len(wsteps) - 1)]
                    take_step(chessboard, current_player, step)
                # print(step)
                current_player = - current_player
                count = count + 1
            evaluations.append(diff(chessboard))
        print("Evaluations: " + str(evaluations))

        parents = []
        values = []
        total = 0
        for i in range(len(genes)):
            if evaluations[i] > 0:
                parents.append(genes[i])
                values.append(evaluations[i])
                total = total + evaluations[i]
        # for parent in parents:
        #     print(parent)
        # print(values)
        # print(total)
        p = [values[0]]  # 权重
        for index in range(1, len(values)):
            p.append(p[index - 1] + values[index])
        # print(p)

        dna = []
        for i in range(10):
            for number in range(35):
                r = random.randint(0, total)
                for j in range(len(p)):
                    if r <= p[j]:
                        dna.append(parents[j][number])
                        number = number + 1
                        break
            parents.append(dna)
            dna = []
        # print(parents)
        genes = parents

