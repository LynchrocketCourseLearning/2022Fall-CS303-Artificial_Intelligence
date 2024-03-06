import numpy as np
import random
import time
import timeout_decorator
# from numba import njit

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

DIRECTIONS = np.array([[0, 1], [0, -1], [1, 0], [-1, 0],
                      [1, 1], [1, -1], [-1, 1], [-1, -1]])

VAL_MAP = np.array([[500, -25, 10, 5, 5, 10, -25, 500],
                    [-25, -45, 1, 1, 1, 1, -45, -25],
                    [10, 1, 3, 2, 2, 3, 1, 10],
                    [5, 1, 2, 1, 1, 2, 1, 5],
                    [5, 1, 2, 1, 1, 2, 1, 5],
                    [10, 1, 3, 2, 2, 3, 1, 10],
                    [-25, -45, 1, 1, 1, 1, -45, -25],
                    [500, -25, 10, 5, 5, 10, -25, 500]])

WEIGHT = [2, 15, 10]

# VAL_MAP = np.array([[500, -25, -25, 500],
#                     [-25, -45, -45, -25],
#                     [-25, -45, -45, -25],
#                     [500, -25, -25, 500]])

############## utils ######################

# @njit(inline='always')


def in_chessboard(idx) -> bool:
    return idx[0] >= 0 and idx[0] < chessboard_size and idx[1] >= 0 and idx[1] < chessboard_size

# @njit(cache=True)


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

# @njit(cache=True)


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

# @njit(cache=True, inline='always')


def is_terminal(chessboard: np.ndarray, color) -> bool:
    """
        终局判定，双方均无子可下
    """
    return len(valid_moves(chessboard, color)) == 0 and len(valid_moves(chessboard, -color)) == 0

# @njit(cache=True, inline='always')


def get_winner(chessboard: np.ndarray):
    black, white = np.count_nonzero(
        chessboard == COLOR_BLACK), np.count_nonzero(chessboard == COLOR_WHITE)
    return -1 if white > black else 1

# @njit(cache=True, inline='always')


def final_point(chessboard: np.ndarray, color) -> int:
    """
        终局评分。搜索到终局用这个，否则用下面的evaluation()
        若color==-1，黑比白多时为负；若color==1，白比黑多为负。
    Params:
        chessboard: np.ndarray,      棋盘
        color: int                   棋子颜色
    Return:
        int                          终局评分    
    """
    black, white = np.count_nonzero(
        chessboard == COLOR_BLACK), np.count_nonzero(chessboard == COLOR_WHITE)
    return color*(black - white)


# @njit(inline='always')


def map_value(chessboard: np.ndarray, color) -> int:
    return color*sum(sum(chessboard*VAL_MAP))

# @njit(inline='always')


def evaluation(chessboard: np.ndarray, color) -> int:
    value = map_value(chessboard, color)  # 位置权值
    # my_moves, other_moves = valid_moves(
    #     chessboard, color), valid_moves(chessboard, -color)  # 行动力
    # stables = stable_value(chessboard, color)  # 稳定子
    # 反黑白棋要取反
    # return -(WEIGHT[0]*value + WEIGHT[1]*(len(my_moves)-len(other_moves)) + WEIGHT[2]*sum(stables))
    # return -(WEIGHT[0]*value + WEIGHT[1]*(len(my_moves)-len(other_moves)))
    return -value

############### minimax a-b pruning ##################


def minimize(chessboard: np.ndarray, color, alpha, beta, depth):
    if is_terminal(chessboard, color):
        return None, final_point(chessboard, color)

    minMove, minUtility = None, np.inf
    moves = valid_moves(chessboard, color)

    # 没得走过一手
    if len(moves) == 0:
        return maximize(chessboard, -color, alpha, beta, depth-1)

    moves.sort(key=lambda mv: evaluation(
        flip_chess(chessboard, color, mv), color))
    if depth <= 1:
        return moves[0], evaluation(flip_chess(chessboard, color, moves[0]), color)

    for move in moves:
        new_chessboard = flip_chess(chessboard, color, move)
        _, utility = maximize(new_chessboard, -color, alpha, beta, depth-1)
        if utility < minUtility:
            minMove, minUtility = move, utility
        if minUtility <= alpha:
            break
        if minUtility < beta:
            beta = minUtility

    return minMove, minUtility


def maximize(chessboard: np.ndarray, color, alpha, beta, depth):
    if is_terminal(chessboard, color):
        return None, final_point(chessboard, color)

    maxMove, maxUtility = None, -np.inf
    moves = valid_moves(chessboard, color)

    # 没得走过一手
    if len(moves) == 0:
        return minimize(chessboard, -color, alpha, beta, depth-1)

    moves.sort(key=lambda mv: evaluation(flip_chess(
        chessboard, color, mv), color), reverse=True)
    if depth <= 1:
        return moves[0], evaluation(flip_chess(chessboard, color, moves[0]), color)

    for move in moves:
        new_chessboard = flip_chess(chessboard, color, move)
        _, utility = minimize(new_chessboard, -color, alpha, beta, depth-1)
        if utility > maxUtility:
            maxMove, maxUtility = move, utility
        if maxUtility >= beta:
            break
        if maxUtility > alpha:
            alpha = maxUtility

    return maxMove, maxUtility


# @timeout_decorator.timeout(4.7)
def decision(chessboard: np.ndarray, color, depth):
    # move, utility = maximize(chessboard, color, -np.inf, np.inf, depth)
    # return move, utility
    moves = valid_moves(chessboard, color)
    if len(moves) > 0:
        return maximize(chessboard, color, -np.inf, np.inf, depth)
    else:
        return None, None

###################################


class AI(object):
    def __init__(self, board_size, color, time_out):
        global chessboard_size
        self.chessboard_size = board_size
        chessboard_size = self.chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.first_move = True
        self.depth = 4
        # self.turn = 1

    def go(self, chessboard: np.ndarray):
        self.candidate_list.clear()
        if self.first_move:
            first_moves = valid_moves(chessboard, self.color)
            self.candidate_list = list(
                map(lambda idx: (idx[0], idx[1]), first_moves))
            self.first_move = False
        else:
            try:
                pos, _ = decision(chessboard, self.color, self.depth)
                if pos is not None:
                    self.candidate_list.append((pos[0], pos[1]))
                else:
                    self.candidate_list.clear()
                # self.turn += 1
                # if self.turn == 10:
                #     self.depth += 2
            except timeout_decorator.timeout_decorator.TimeoutError:
                # 随机放
                self.depth -= 2
                moves = valid_moves(chessboard, self.color)
                if len(moves) > 0:
                    moves.sort(key=lambda mv: evaluation(flip_chess(
                        chessboard, self.color, mv), self.color), reverse=True)
                    self.candidate_list.append((moves[0][0], moves[0][1]))
                else:
                    self.candidate_list.clear()
                # self.turn += 1

            # pos, _ = decision(chessboard, self.color, 4)
            # if pos is not None:
            #     self.candidate_list.append((pos[0], pos[1]))
            # else:
            #     self.candidate_list.clear()


if __name__ == '__main__':
    ai = AI(8, -1, 0)
    board = np.array([[0,  0,  0, 0, 0,  0,  0, 0],
                      [0,  0, 0, -1, -1, -1, -1, 0],
                      [1, -1, -1, -1, -1, -1, -1, -1],
                      [1,  1, -1,  1, 1, 1,  1, -1],
                      [1, 1,  1, -1, 1, 1,  1, 1],
                      [1, 1, -1, -1, -1,  1, 1, 1],
                      [1, 1, -1, -1, 1,  1, 1,  1],
                      [1, 1, 1, 1, 1, 1, -1, 1]])

    start = time.time()
    ai.first_move = False
    ai.go(board)
    print(ai.candidate_list)
    print(time.time()-start)

# if __name__ == '__main__':
#     ai_1 = AI(8, -1, 0)
#     ai_2 = AI(8, 1, 0)
#     # board = np.array([
#     #     [0, 0, 0, 0],
#     #     [0, 1, -1, 0],
#     #     [0, -1, 1, 0],
#     #     [0, 0, 0, 0]
#     # ])
#     board = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, -1, 1, 0, 0, 0],
#         [0, 0, 0, 1, -1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0]
#     ])
#     start = time.time()
#     while True:
#         # start = time.time()
#         ai_1.go(board)
#         move = ai_1.candidate_list[-1]
#         board = flip_chess(board, -1, move)
#         # print(time.time()-start)
#         # print(board)

#         # start = time.time()
#         ai_2.go(board)
#         move = ai_2.candidate_list[-1]
#         board = flip_chess(board, 1, move)
#         # print(time.time()-start)
#         # print(board)

#         if is_terminal(board, 1):
#             break
#     print(time.time()-start)
#     print(board)
#     print(get_winner(board))
