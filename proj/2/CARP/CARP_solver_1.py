import argparse
import numpy as np
import copy
import multiprocessing as mp
import os

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

graph, distance = None, None

def main(args: argparse.ArgumentParser):
    demands = parse_data(args)
    routes, total_cost = init_population(demands)
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
        routes: list of route, i.e., list[list[tuple]]
        total_cost:
    '''
    global distance
    
    total_cost: float = 0.0 # 该routes的总cost
    routes = [] # 存所有的route
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

            # d_o 是选择的distance，u_o是选择的edge
            d_o, u_o = np.inf, -1
            choose_term = -1
            for u in free_not_exceed_capacity:
                if distance[i, u[IDX_BEGIN_NODE]] < d_o:
                    d_o = distance[i, u[IDX_BEGIN_NODE]]
                    u_o = u
                elif distance[i, u[IDX_BEGIN_NODE]] == d_o:
                    if rule == 1 or rule == 2 or rule == 5:
                        term = distance[i, u[IDX_BEGIN_NODE]]
                        maximize = True if rule == 1 else (route_load <= CAPACITY // 2 if rule == 5 else False)
                    elif rule == 3 or rule == 4:
                        term = u[IDX_DEMAND] / distance[i, u[IDX_BEGIN_NODE]] if distance[i, u[IDX_BEGIN_NODE]] != 0 else np.inf
                        maximize = True if rule == 3 else False
                    else:
                        pass
                        
                    if (maximize and choose_term <= term) or (not maximize and choose_term > term):
                        u_o = u
                        choose_term = term

            if d_o == np.inf:
                # 此时 u_o == -1，即没选到合适的edge
                break

            route.append(u_o)
            i = u_o[IDX_END_NODE]
            route_load += u_o[IDX_DEMAND]
            route_cost += d_o + u_o[IDX_COST]
            free.remove(u_o)
            free.remove((u_o[IDX_END_NODE], u_o[IDX_BEGIN_NODE], u_o[IDX_COST], u_o[IDX_DEMAND]))

        route_cost += distance[i, DEPOT]
        total_cost += route_cost
        routes.append(route)

    return routes, total_cost

def flip(routes: list, total_cost: float):
    global distance
    cost = total_cost
    for i in range(len(routes)): # 遍历所有的route
        route = routes[i]
        for j in range(len(route)): # 对每个route经过的edge
            st = DEPOT if j == 0 else route[j-1][IDX_END_NODE]
            ed = DEPOT if j == len(route)-1 else route[j+1][IDX_BEGIN_NODE]
            
            diff = distance[ed, route[j][IDX_END_NODE]] + distance[route[j][IDX_BEGIN_NODE], st] - distance[st, route[j][IDX_BEGIN_NODE]] - distance[route[j][IDX_END_NODE], ed]
            if diff < 0:
                cost += diff
                route[j] = [(route[i][IDX_END_NODE], route[i][IDX_BEGIN_NODE], route[i][IDX_DEMAND], route[i][IDX_DEMAND])]
    
    return cost

def init_population(demands: np.ndarray):
    population = []
    for rule in [1, 2, 3, 4, 5]:
        free = copy.deepcopy(demands)
        routes, total_cost = path_scanning(free, rule)
        population.append((routes, total_cost))
    
    population_sorted = sorted(population, key=lambda x: x[-1]) # 按total_cost递增排
    return population_sorted[0]


def get_res(routes: list, total_cost) -> str:
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
