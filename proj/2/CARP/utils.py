import copy
def two_opt(routes: list, capacity_cost: list, single: bool):
    total_cost = 0
    for i, (route, cc) in enumerate(zip(routes, capacity_cost)):
        choose_route = route
        cost = cc[1]
        for st in range(len(route)-1):
            for ed in range(st+2, len(route)):
                copy_route = copy.deepcopy(route)
                new_cost = 1
                if new_cost < cost:
                    choose_route = copy_route
                    cost = new_cost
                    capacity_cost[i] = (capacity_cost[i][0], cost)
        routes[i] = choose_route
        total_cost += cost
    return total_cost