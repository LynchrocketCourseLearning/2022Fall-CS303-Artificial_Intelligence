from typing import Union
import numpy as np
import random
import time
import timeout_decorator

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)

DIRECTIONS = np.array([[0,1], [0,-1], [1,0], [-1,0], [1,1], [1,-1], [-1,1], [-1,-1]])

# VAL_MAP = np.array([[500, -25, 10, 5, 5, 10, -25, 500],
#                     [-25, -45, 1, 1, 1, 1, -45, -25],
#                     [10, 1, 3, 2, 2, 3, 1, 10],
#                     [5, 1, 2, 1, 1, 2, 1, 5],
#                     [5, 1, 2, 1, 1, 2, 1, 5],
#                     [10, 1, 3, 2, 2, 3, 1, 10],
#                     [-25, -45, 1, 1, 1, 1, -45, -25],
#                     [500, -25, 10, 5, 5, 10, -25, 500]])

VAL_MAP = np.array([
    [1,5,3,3,3,3,5,1],
    [5,5,4,4,4,4,5,5],
    [3,4,2,2,2,2,4,3],
    [3,4,2,0,0,2,4,3],
    [3,4,2,0,0,2,4,3],
    [3,4,2,2,2,2,4,3],
    [5,5,4,4,4,4,5,5],
    [1,5,3,3,3,3,5,1],
])

######  class  ############

class Node:
    def __init__(self, chessboard, color, parent=None, action=None):
        self.chessboard: np.ndarray = chessboard
        self.color: int = color
        self.parent: Union[Node, None] = parent
        self.children: list[Node] = []
        self.Q: int = 0 # 价值
        self.N: int = 0 # 访问次数

        # 用于修改后的UCB
        # self.win_times: int = 0 # 总模拟的赢的次数
        # self.total_times: int = 0 # 总模拟的次数

        self.action: Union[list[int], None] = action # 自父节点到该节点的action
        self.moves: dict[tuple, Node] = {}
        # self.map_of_nodes: dict[int, Node] = {} # 对所有子节点的映射

######  utils  ############

def valid_moves(chessboard: np.ndarray, color):
    """
        找到所有可能的有效下棋位置
    :params
        chessboard: np.ndarray,      棋盘
        color: int                   棋子颜色
    :return
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

def terminal_reward(chessboard, color) -> int:
    """
        终局计算
    """
    own = np.count_nonzero(chessboard == color)
    other = np.count_nonzero(chessboard == -color)
    if own < other:
        return 1
    elif other > own:
        return -1
    else:
        return 0

def evaluation(chessboard: np.ndarray, color) -> int:
    value = color*sum(sum(chessboard*VAL_MAP))
    own, other = np.count_nonzero(
        chessboard == color), np.count_nonzero(chessboard == -color)
    # return value + 5*(own - other)
    return value

######  MCTS  #############

# 原UCB1
def UCT(node: Node, C=1.41) -> float:
    if node.N == 0:
        return float('inf')
    if node.parent is None:
        return 0
    return node.Q / node.N + C * np.sqrt(2 * np.log(node.parent.N) / node.N)

# 修改后的UCB1
# def UCT(node: Node, C=1.414, b=10) -> float:
#     if node.N == 0 or node.total_times == 0:
#         return float('inf')
#     if node.parent is None:
#         return 0
#     beta = node.total_times/(node.N+node.total_times+4*b*b*node.N*node.total_times)
#     return (1-beta)*node.Q / node.N + beta*node.win_times/node.total_times + C * np.sqrt(2 * np.log(node.parent.N) / node.N)


# 已经inline了，弃用
# def best_child(node: Node) -> Node:
#     return max(node.children, key=UCT)

def tree_policy(node: Node) -> Node:
    leaf = node
    # select
    while len(leaf.children) > 0:
        # leaf = best_child(leaf)
        leaf = max(leaf.children, key=UCT) # 手动inline
    
    # expand
    # if leaf.N != 0:
    #     moves = valid_moves(leaf.chessboard, leaf.color)
    #     # 还需要补充，若len(moves)==0需要跳一步
    #     if len(moves) == 0:
    #         child = Node(leaf.chessboard, -leaf.color, parent=leaf, action=None)
    #         leaf.children.append(child)
    #     for pos in moves:
    #         new_chessboard = flip_chess(leaf.chessboard, leaf.color, pos)
    #         child = Node(new_chessboard, -leaf.color, parent=leaf, action=pos)
    #         leaf.children.append(child)
    #     return leaf.children[0]
    #
    # return leaf

    moves = valid_moves(leaf.chessboard, leaf.color)
    if len(moves) == 0:
        child = Node(leaf.chessboard, -leaf.color, parent=leaf, action=None)
        leaf.children.append(child)
    else:
        for pos in moves:
            new_chessboard = flip_chess(leaf.chessboard, leaf.color, pos)
            child = Node(new_chessboard, -leaf.color, parent=leaf, action=pos)
            leaf.children.append(child)

    return leaf

def default_policy(node: Node):
    """
        轮流对弈, 模拟 (simulation)\n
        终局判断。如果该颜色没有合法落子位置，则轮到对方落子，直到有合法落子位置为止。若两方均过一手，则结束。
    """
    new_chessboard = node.chessboard.copy()
    color = node.color
    level = 0 # 模拟层数

    moves = valid_moves(new_chessboard, color)
    # 两次循环保证了过两手的情况，就不用过手计数了
    while len(moves) > 0:
        while len(moves) > 0:
            # 模拟策略1：随机
            pos = random.choice(moves)

            # 模拟策略2：棋盘权值得分
            # if len(moves) > 1:
            #     if color == COLOR_BLACK:
            #         moves.sort(key=lambda mv: evaluation(flip_chess(new_chessboard, color, mv), color), reverse=True)
            #     else:
            #         moves.sort(key=lambda mv: evaluation(flip_chess(new_chessboard, color, mv), color))
            # pos = moves[0]

            # 模拟策略3：Roxanne优先级表
            # moves.sort(key=lambda mv: color*VAL_MAP[mv[0], mv[1]], reverse=True)
            pos = moves[0]
            new_chessboard = flip_chess(new_chessboard, color, pos)
            color = -color
            level += 1
            # if level >= 30:
            #     break
            moves = valid_moves(new_chessboard, color)
        color = -color
        level += 1
        moves = valid_moves(new_chessboard, color)

    return terminal_reward(new_chessboard, color)

def backup(node: Node, reward: int) -> None:
    """
        回溯传播 (backpropagation)
    """
    cur = node
    while cur:
        cur.N += 1
        cur.Q += reward
        
        # cur.total_times += 1
        # if reward > 0:
        #     cur.win_times += 1

        reward = -reward
        cur = cur.parent

# 弃用了
# def MCTS(node: Node):
#     computational_budget = 100
#     for _ in range(computational_budget):
#         expend_node = tree_policy(node)
#         # print(f'{i}-th tree over')
#         node_reward = default_policy(expend_node)
#         # print(f'{i}-th default over')
#         backup(expend_node, node_reward)
#         # print(f'{i}-th backup over')
#
#     return best_child(node).action
    
class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.first_move = True
        self.level: int = 0

        # self.actions: dict[tuple, Node] = {}

    def go(self, chessboard: np.ndarray):
        self.candidate_list.clear()
        node = Node(chessboard, self.color)
        self.level += 1

        if self.first_move:
            first_moves = valid_moves(chessboard, self.color)
            self.candidate_list = list(map(lambda idx: (idx[0], idx[1]), first_moves))
            self.first_move = False
        else:
            try:
                MCTS(node)
            except timeout_decorator.timeout_decorator.TimeoutError:
                if len(node.moves) <= 0:
                    self.candidate_list.clear()
                else:
                    self.candidate_list = sorted(node.moves.keys(), key=lambda x: UCT(node.moves[x], C=0))
                    # self.candidate_list = sorted(node.moves.keys(), key=lambda x: evaluation(node.moves[x].chessboard, node.color))
            
@timeout_decorator.timeout(4.8)
def MCTS(node: Node):
    moves = valid_moves(node.chessboard, node.color)
    if len(moves) <= 0:
        return
    else:
        for move in moves:
            new_chessboard = flip_chess(node.chessboard, node.color, move)
            child = Node(new_chessboard, -node.color, node, move)
            node.children.append(child)
            node.moves[(move[0], move[1])] = child
        # 先弄好第一层
        for child in node.children:
            node_reward = default_policy(child)
            backup(child, node_reward)

        while True:
            expend_node = tree_policy(node)
            for child in expend_node.children:
                node_reward = default_policy(child)
                backup(child, node_reward)

# if __name__ == '__main__':
#     ai = AI(8, -1, 0)
#     board = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0], 
#         [0, 0, 0, 0, 0, 0, 0, 0], 
#         [0, 0, -1, 0, 0, 0, -1, 0], 
#         [0, 0, -1, -1, -1, 1, 1, 1],
#         [-1, -1, -1, -1, 1, 1, 1, 1], 
#         [-1, 1, 1, 1, 1, 1, -1, 1], 
#         [-1, 1, 1, 1, 1, 1, 1, 1], 
#         [-1, 1, 1, 1, 1, 1, 1, 1]
#     ])
#     start = time.time()
#     ai.first_move = False
#     ai.go(board)
#     print(ai.candidate_list)
#     print(time.time()-start)

#     while True:
#         start = time.time()
#         ai.go(board)
#         move = ai.candidate_list[-1]
#         flip_chess(board, -1, move)
#         print(time.time()-start)
#         print(board)
#         user_move = input('move: ').split(',')
#         flip_chess(board, 1, [int(user_move[0]), int(user_move[1])])

        