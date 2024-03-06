import argparse
import numpy as np
import copy
import time
import random

parser = argparse.ArgumentParser()

parser.add_argument('instance', type=str, default='./CARP_samples/egl-e1-A.dat')
parser.add_argument('-t', '--termination', type=int, default=60)
parser.add_argument('-s', '--seed', type=int, default=100)

# constants
NAME = 'egl-e1-A'
VERTICES = DEPOT = REQUIRED_EDGES = NON_REQUIRED_EDGES = VEHICLES = CAPACITY = TOTAL_COST_OF_REQUIRED_EDGES = 0
INSTANCE = './CARP_samples/gdb1.dat'
TERMINATION = 60
SEED = 100

# indexes
IDX_BEGIN_NODE, IDX_END_NODE, IDX_COST, IDX_DEMAND = 0, 1, 2, 3

# graphs
graph: np.ndarray = None
distance: np.ndarray = None

# timer
st_time: float = time.time()

# Simulated Annealing
T = 10000 # 初始温度
K = 0.95 # 衰减参数
L = 200 # 马尔科夫链长度

def main(args: argparse.ArgumentParser):
    demands = parse_data(args)
    random.seed(SEED)
    routes, capacity_cost, total_cost = simulated_annealing(demands)
    print(get_res(routes, total_cost))

def parse_data(args: argparse.ArgumentParser):
    global NAME
    global VERTICES, DEPOT, REQUIRED_EDGES, NON_REQUIRED_EDGES, VEHICLES, CAPACITY, TOTAL_COST_OF_REQUIRED_EDGES
    global INSTANCE, TERMINATION, SEED
    global graph, distance

    INSTANCE = args.instance
    TERMINATION = args.termination
    SEED = args.seed

    with open(INSTANCE, 'r') as file:
        NAME = file.readline().split(' : ')[1].strip()
        VERTICES = int(file.readline().split(' : ')[1])
        DEPOT = int(file.readline().split(' : ')[1])
        REQUIRED_EDGES = int(file.readline().split(' : ')[1])
        NON_REQUIRED_EDGES = int(file.readline().split(' : ')[1])
        VEHICLES = int(file.readline().split(' : ')[1])
        CAPACITY = int(file.readline().split(' : ')[1])
        TOTAL_COST_OF_REQUIRED_EDGES = int(file.readline().split(' : ')[1])

        file.readline()

        graph = np.full((VERTICES + 1, VERTICES + 1), np.inf, dtype=np.float64)
        demands = []
        while (line := file.readline()) != 'END':
            node1, node2, cost, demand = list(map(int, line.split()))
            graph[node1, node2] = cost
            graph[node2, node1] = cost

            if demand != 0:
                # 反序一并存了，偶数下标是正序的，奇数下标是反序的
                demands.append((node1, node2, cost, demand))
                demands.append((node2, node1, cost, demand))

    distance = floyd(graph)

    return demands


def floyd(graph: np.ndarray) -> np.ndarray:
    n = len(graph)
    distance = copy.deepcopy(graph)
    for k in range(1, n):
        distance[k, k] = 0
        for i in range(1, n):
            for j in range(1, n):
                distance[i, j] = min(distance[i, j], distance[i, k] + distance[k, j])

    return distance


def path_scanning(free: list, rule: int):
    '''
    :param
        free: list of tuple(BEGIN_NODE, END_NODE, COST, DEMAND)
        rule:   1 for maximize the distance from the task to the depot;\n
                2 for minimize the distance from the task to the depot;\n
                3 for maximize the term dem(t)/sc(t);\n
                4 for minimize the term dem(t)/sc(t);\n
                5 for use rule 1) if the vehicle is less than half- full, otherwise use rule 2)
                
    :return
        routes: list of the edge list, i.e., list[list[tuple]]
        capacity_cost: list of list in [reamin_capacity, route_cost]
        total_cost: float as sum of route_cost
    '''
    global distance
    global DEPOT, CAPACITY
    global IDX_BEGIN_NODE, IDX_END_NODE, IDX_COST, IDX_DEMAND
    
    total_cost: float = 0.0 # 该routes的总cost
    routes = [] # 存所有的route
    capacity_cost = [] # 每个route之后的capacity余量和cost
    # k = 0 # counter
    while len(free) != 0:
        # k += 1
        # print(k)
        route = []
        route_load, route_cost = 0, 0
        i = DEPOT  # starts from depot point
        while True:
            free_not_exceed_capacity = [ele for ele in free if route_load + ele[IDX_DEMAND] <= CAPACITY]
            if len(free_not_exceed_capacity) == 0:
                break

            # d_o 是选择的distance，u_o是选择的edge, choose_term用来区分5种规则
            d_o, u_o = np.inf, -1
            choose_term = -1
            # 随机取的时候要用到，随机的时候不用choose_term
            random_choose_routes = []
            for u in free_not_exceed_capacity:
                if distance[i, u[IDX_BEGIN_NODE]] < d_o:
                    d_o = distance[i, u[IDX_BEGIN_NODE]]
                    u_o = u
                    random_choose_routes.append(u)
                elif distance[i, u[IDX_BEGIN_NODE]] == d_o:
                    maximize = False
                    if rule == 1 or rule == 2 or rule == 5:
                        term = distance[i, u[IDX_BEGIN_NODE]]
                        maximize = True if rule == 1 else (route_load <= CAPACITY // 2 if rule == 5 else False)
                    elif rule == 3 or rule == 4:
                        term = u[IDX_DEMAND] / distance[i, u[IDX_BEGIN_NODE]] if distance[i, u[IDX_BEGIN_NODE]] != 0 else np.inf
                        maximize = True if rule == 3 else False
                    else: # 随机取
                        random_choose_routes.append(u)
                        continue
                        
                    if (maximize and choose_term <= term) or (not maximize and choose_term > term):
                        u_o = u
                        choose_term = term

            if d_o == np.inf:
                # 此时 u_o == -1，即没选到合适的edge
                break
            
            if rule not in [1, 2, 3, 4, 5]:
                u_o = random_choose_routes[random.randint(0, len(random_choose_routes)-1)]
                
            route.append(u_o)
            i = u_o[IDX_END_NODE]
            route_load += u_o[IDX_DEMAND]
            route_cost += d_o + u_o[IDX_COST]
            free.remove(u_o)
            free.remove((u_o[IDX_END_NODE], u_o[IDX_BEGIN_NODE], u_o[IDX_COST], u_o[IDX_DEMAND]))
            
        route_cost += distance[i, DEPOT]
        total_cost += route_cost
        routes.append(route)
        capacity_cost.append([CAPACITY-route_load, route_cost])

    return routes, capacity_cost, total_cost

def flip(routes: list, capacity_cost: list, total_cost: float):
    '''
    :param
        routes: list of the edge list, i.e., list[list[tuple]],
                where edge is in tuple(BEGIN_NODE, END_NODE, COST, DEMAND)
        capacity_cost: list of list in [reamin_capacity, route_cost]
        total_cost: float as sum of route_cost
                
    :return
        new_routes: the new routes
        new_capacity_cost: the new capacity_cost
        new_total_cost: the new total cost of routes after flip
    '''
    global distance
    global DEPOT
    global IDX_BEGIN_NODE, IDX_END_NODE, IDX_COST, IDX_DEMAND
    
    new_routes = copy.deepcopy(routes)
    new_capacity_cost = copy.deepcopy(capacity_cost)
    new_total_cost = total_cost
    for i in range(len(new_routes)): # 遍历所有的route
        route = new_routes[i]
        for j in range(len(route)): # 对每个route经过的edge
            st = DEPOT if j == 0 else route[j-1][IDX_END_NODE]
            ed = DEPOT if j == len(route)-1 else route[j+1][IDX_BEGIN_NODE]
            
            diff = distance[ed, route[j][IDX_END_NODE]] + distance[route[j][IDX_BEGIN_NODE], st] - distance[st, route[j][IDX_BEGIN_NODE]] - distance[route[j][IDX_END_NODE], ed]
            if diff < 0:
                new_total_cost += diff
                route[j] = [(route[i][IDX_END_NODE], route[i][IDX_BEGIN_NODE], route[i][IDX_DEMAND], route[i][IDX_DEMAND])]
                new_routes[i][j] = route[j]
                new_capacity_cost[i][1] += diff
    
    return new_routes, new_capacity_cost, new_total_cost

def swap(routes: list, capacity_cost: list, total_cost: float):
    '''
    :param
        routes: list of the edge list, i.e., list[list[tuple]],
                where edge is in tuple(BEGIN_NODE, END_NODE, COST, DEMAND)
        capacity_cost: list of list in [reamin_capacity, route_cost]
        total_cost: float as sum of route_cost
                
    :return
        new_routes: the new routes
        new_capacity_cost: the new capacity_cost
        new_total_cost: the new total cost of routes after flip
    '''
    global distance
    global DEPOT, CAPACITY
    global IDX_BEGIN_NODE, IDX_END_NODE, IDX_COST, IDX_DEMAND
    
    new_routes = copy.deepcopy(routes)
    new_capacity_cost = copy.deepcopy(capacity_cost)
    new_total_cost = total_cost
    
    while True:
        route1_idx = random.randint(0, len(new_routes)-1)
        route2_idx = random.randint(0, len(new_routes)-1)
        if route1_idx != route2_idx:
            break
    route1 = new_routes[route1_idx]
    route2 = new_routes[route2_idx]
    
    # for edge1_idx in range(len(route1)-1):
    #     for edge2_idx in range(len(route2)-1):
    n = len(route1)+len(route2)
    for _ in range(n):
        # 控制时间
        remain_time = TERMINATION - (time.time() - st_time)
        if remain_time < 1.0:
            break
        edge1_idx = random.randint(0, len(route1)-1)
        edge2_idx = random.randint(0, len(route2)-1)
        # 要交换的两条边
        edge1 = route1[edge1_idx]
        edge2 = route2[edge2_idx]
            
        p_st = DEPOT if edge1_idx == 0 else route1[edge1_idx-1][IDX_END_NODE]
        p_ed = DEPOT if edge1_idx == len(route1)-1 else route1[edge1_idx+1][IDX_BEGIN_NODE]
            
        q_st = DEPOT if edge2_idx == 0 else route2[edge2_idx-1][IDX_END_NODE]
        q_ed = DEPOT if edge2_idx == len(route2)-1 else route2[edge2_idx+1][IDX_BEGIN_NODE]
        
        capacity1 = CAPACITY - new_capacity_cost[route1_idx][0] - edge2[IDX_DEMAND] + edge1[IDX_DEMAND]
        capacity2 = CAPACITY - new_capacity_cost[route2_idx][0] - edge1[IDX_DEMAND] + edge2[IDX_DEMAND]
        
        if capacity1 <= CAPACITY and capacity2 <= CAPACITY:
            break
        
    # 交换后分别的cost差
    diff1 = -distance[p_st, edge1[IDX_BEGIN_NODE]]-distance[edge1[IDX_END_NODE], p_ed]-edge1[IDX_COST]\
            +distance[p_st, edge2[IDX_BEGIN_NODE]]+distance[edge2[IDX_END_NODE], p_ed]+edge2[IDX_COST]
    diff2 = -distance[q_st, edge2[IDX_BEGIN_NODE]]-distance[edge2[IDX_END_NODE], q_ed]-edge2[IDX_COST]\
            +distance[q_st, edge1[IDX_BEGIN_NODE]]+distance[edge1[IDX_END_NODE], q_ed]+edge1[IDX_COST]
            
    if new_total_cost + diff1 + diff2 < new_total_cost:
        new_total_cost += diff1 + diff2
        new_route1 = route1[:edge1_idx]+[edge2]+route1[edge1_idx+1:]
        new_route2 = route1[:edge2_idx]+[edge1]+route2[edge2_idx+1:]
        new_capacity_cost[route1_idx][0] = CAPACITY-capacity1
        new_capacity_cost[route2_idx][0] = CAPACITY-capacity2
        new_capacity_cost[route1_idx][1] += diff1
        new_capacity_cost[route2_idx][1] += diff2
        new_routes[route1_idx] = new_route1
        new_routes[route2_idx] = new_route2
    
    return new_routes, new_capacity_cost, new_total_cost

def init_solution(demands: np.ndarray):
    '''
    initialize the solution
    :param
        demands: the tasks
        
    :return

    '''
    solution = []
    for rule in [1, 2, 3, 4, 5]: # 五种规则
        free = copy.deepcopy(demands)
        routes, capacity_cost, total_cost = path_scanning(free, rule)
        solution.append((routes, capacity_cost, total_cost))
    for _ in range(10): # 随机，rule给除1，2，3，4，5以外的值
        free = copy.deepcopy(demands)
        routes, capacity_cost, total_cost = path_scanning(free, 0)
        solution.append((routes, capacity_cost, total_cost))
    
    solution_sorted = sorted(solution, key=lambda x: x[-1]) # 按total_cost递增排
    return solution_sorted[0] # 有最小的total_cost

def new_solution(routes: list, capacity_cost: list, total_cost: float):
    '''
    generate new solution
    :param
        routes: list of the edge list, i.e., list[list[tuple]],
                where edge is in tuple(BEGIN_NODE, END_NODE, COST, DEMAND)
        capacity_cost: list of list in [reamin_capacity, route_cost]
        total_cost: float as sum of route_cost
                
    :return
    
    '''
    # new_solutions = [flip(routes, capacity_cost, total_cost)]
    new_solutions = [flip(routes, capacity_cost, total_cost), swap(routes, capacity_cost, total_cost)]
    new_solutions_sorted = sorted(new_solutions, key=lambda x: x[-1])
    return new_solutions_sorted[0]

def simulated_annealing(demands: np.ndarray):
    '''
    :param
        demands: the tasks
                
    :return
        current: the best solution
    '''
    global TERMINATION, st_time
    remain_time = TERMINATION - (time.time() - st_time)
    
    global T, K, L
    
    temperature = T
    # init_routes, init_capacity_cost, init_total_cost
    current = init_solution(demands)
    while temperature > 0.1:
        temperature *= K

        for _ in range(L):
            next = new_solution(*current)
            diff = next[-1] - current[-1] # 若为正，则next的价值更低；否则next的价值更高
            if diff < 0:
                current = next
            elif random.random() < np.exp(-diff/temperature):
                current = next
        
        # 控制时间
        remain_time = TERMINATION - (time.time() - st_time)
        if remain_time < 1.0:
            break 
    
    return current

def get_res(routes: list, total_cost: float) -> str:
    '''
    turn the routes and total_cost to the required answer
    :param
        routes: list of the edge list, i.e., list[list[tuple]],
                where edge is in tuple(BEGIN_NODE, END_NODE, COST, DEMAND)
        total_cost: float as sum of route_cost
    
    :return
        a `str` of answer
    '''
    res = []
    for route in routes:
        res.append(0)
        for edge in route:
            res.append((edge[IDX_BEGIN_NODE], edge[IDX_END_NODE]))
        res.append(0)
    return 's ' + ','.join([str(x) for x in res]) + '\nq ' + str(int(total_cost))
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
