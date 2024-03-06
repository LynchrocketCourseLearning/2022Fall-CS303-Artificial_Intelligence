## 一些有用的函数

import numpy as np
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
DIRECTIONS = np.array([[0,1], [0,-1], [1,0], [-1,0], [1,1], [1,-1], [-1,1], [-1,-1]])


def valid_moves(chessboard: np.ndarray, color):
    """
        找到所有可能的有效下棋位置
    Params:
        chessboard: np.ndarray,      棋盘
        color: int                   棋子颜色
    Return:
        moves: List[List[int, int]]  所有可能的有效下棋位置
    """
    moves = []
    chesses = np.argwhere((chessboard == color))
    in_chessboard = (lambda idx: (idx[0] in range(len(chessboard)) and idx[1] in range(len(chessboard))))
    
    for pos in chesses:
        neighbours = pos + DIRECTIONS
        for i, nei in enumerate(neighbours):
            # flag用来保证越过了其他颜色的棋子
            flag = False
            while in_chessboard(nei):
                if chessboard[nei[0], nei[1]] == -color:
                    flag = True
                elif chessboard[nei[0], nei[1]] == color:
                    flag = False
                    break
                # 找到了可以下子的地方
                elif chessboard[nei[0], nei[1]] == COLOR_NONE:
                    break
                nei += DIRECTIONS[i]
            if flag and in_chessboard(nei):
                moves.append(nei)

    return moves

def flip_chess(chessboard: np.ndarray, color, pos) -> None:
    """
        翻转棋子
    """
    in_chessboard = (lambda idx: (idx[0] in range(len(chessboard)) and idx[1] in range(len(chessboard))))
    
    neighbours = pos + DIRECTIONS
    chess_to_flip = []
    for i, nei in enumerate(neighbours):
        # flag用来保证越过了其他颜色的棋子
        flag = False
        tmp_flip = []
        while in_chessboard(nei):
            if chessboard[nei[0], nei[1]] == COLOR_NONE:
                break
            elif chessboard[nei[0], nei[1]] == -color:
                tmp_flip.append(nei.copy())
                nei += DIRECTIONS[i]
                flag = True
            elif chessboard[nei[0], nei[1]] == color:
                if flag:
                    chess_to_flip.append(tmp_flip)
                break
    
    chessboard[pos[0], pos[1]] = color
    for flip_list in chess_to_flip:
        for chess_pos in flip_list:
            chessboard[chess_pos[0], chess_pos[1]] = color


## 另一个版本

def valid_moves(chessboard: np.ndarray, color):
    """
        找到所有可能的有效下棋位置
    Params:
        chessboard: np.ndarray,      棋盘
        color: int                   棋子颜色
    Return:
        moves: List[List[int, int]]  所有可能的有效下棋位置
    """
    moves = []
    chesses = np.argwhere(chessboard == color)
    none_area = np.argwhere(chessboard == COLOR_NONE)

    for pos in none_area:
        # flag用来保证越过了其他颜色的棋子
        flag = False
        for chess in chesses:
            # 空处和棋盘上己方棋子的距离
            dis = chess - pos
            if dis[0] != 0 and dis[1] != 0:
                if abs(dis[0]) != abs(dis[1]):
                    continue
            times = abs(dis[0]) if dis[0] != 0 else abs(dis[1])
            step = dis // times
            # 不会到达chess的位置，故上界为times
            for i in range(1, times):
                now_pos = i*step + pos
                if chessboard[now_pos[0], now_pos[1]] == -color:
                    flag = True
                else:  # 遇到空处或者提前遇到己方棋子，不算有效
                    flag = False
                    break
            if flag:
                moves.append(pos)
                break

    return moves

def flip_chess(chessboard: np.ndarray, color, pos) -> np.ndarray:
    """
        翻转棋子
    """
    chesses = np.argwhere((chessboard == color))
    chess_to_flip = []
    for chess in chesses:
        dis = chess - pos
        if dis[0] != 0 and dis[1] != 0:
            if abs(dis[0]) != abs(dis[1]):
                continue
        tmp_flip = [pos]
        times = abs(dis[0]) if dis[0] != 0 else abs(dis[1])
        step = dis // times
        # 不会到达chess的位置，故上界为times
        for i in range(1, times):
            now_pos = i*step + pos
            # 提前遇到了己方的棋子
            if chessboard[now_pos[0], now_pos[1]] == color:
                tmp_flip.clear()
                break
            tmp_flip.append(now_pos)
        if len(tmp_flip) > 0:
            chess_to_flip.append(tmp_flip)

    new_chessboard = chessboard.copy()
    for flip_list in chess_to_flip:
        for chess_pos in flip_list:
            new_chessboard[chess_pos[0], chess_pos[1]] = color

    return new_chessboard



# 稳定子
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

    if len(inner_stable) == 0:
        return stable

    for chess in inner_stable_chess:
        if (chess[0], chess[1]) in stable_pos:
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