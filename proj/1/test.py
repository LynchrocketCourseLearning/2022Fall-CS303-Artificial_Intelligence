import numpy as np
COLOR_NONE = 0
chessboard_size = 8


def in_chessboard(idx) -> bool:
    return idx[0] >= 0 and idx[0] < chessboard_size and idx[1] >= 0 and idx[1] < chessboard_size


def stable_value(chessboard: np.ndarray, color):
    stable = [0, 0, 0]  # 角，边，中心。中心的稳定子只计算了部分
    stable_pos = set()  # 角、边的稳定子
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    direc1 = [(1, 0), (0, -1), (0, 1), (-1, 0)]
    direc2 = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    for i in range(4):
        cor: tuple = corners[i]
        # 角
        if chessboard[cor[0], cor[1]] == color:
            stable[0] += 1
            # 边，两个方向
            for j in range(1, 7):
                pos = (cor[0]+j*direc1[i][0], cor[1]+j*direc1[i][1])
                if chessboard[pos[0], pos[1]] == color:
                    stable[1] += 1
                    stable_pos.add(pos)
                else:
                    break
            for j in range(1, 7):
                pos = (cor[0]+j*direc2[i][0], cor[1]+j*direc2[i][1])
                if chessboard[pos[0], pos[1]] == color:
                    stable[1] += 1
                    stable_pos.add(pos)
                else:
                    break
    inner_stable = np.zeros((8, 8), dtype=bool)
    inner_stable[:, np.sum(abs(chessboard), axis=0) == 8] = True  # 列
    inner_stable[np.sum(abs(chessboard), axis=1) == 8, :] &= True  # 行
    # 斜线
    inner_stable_chess = np.argwhere(inner_stable == True)
    diag_direc = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    if len(inner_stable_chess) == 0:
        return stable

    for chess in inner_stable_chess:
        if chess in stable_pos:
            continue
        for dd in diag_direc:
            pos = chess + dd
            flag = False
            while in_chessboard(pos):
                if chessboard[pos[0], pos[1]] == COLOR_NONE:
                    inner_stable[chess[0], chess[1]] = False
                    flag = True
                    break
                pos += dd
            if flag:
                break

    stable[2] = sum(sum(inner_stable))
    return stable


board = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, -1, -1, 0, 0, 0, 0, 0],
    [0, -1, 1, 1, 1, 0, 0, 0],
    [0, -1, 1, -1, -1, 0, 0, 0],
    [0, -1, 1, 0, 0, 1, 0, 0],
    [0, -1, -1, 0, 0, 0, 1, 0],
    [0, -1, 1, 0, 0, 0, 0, -1]
])
stable = stable_value(board, 1)
print(stable)



