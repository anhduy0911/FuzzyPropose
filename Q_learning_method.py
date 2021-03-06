import math
import numpy as np
from scipy.spatial import distance

import Parameter as para
from Node_Method import find_receiver
import Fuzzy_Fix


def q_max_function(q_table, state):
    """
    calculate q max
    :param q_table:
    :param state:
    :return:
    """
    temp = [max(row) if index != state else -float("inf") for index, row in enumerate(q_table)]
    return np.asarray(temp)

def determine_theta_heuristic(network):
    low_bound = (2 * network.node[0].energy_thresh - network.node[0].energy_thresh) / network.node[0].energy_max
    high_bound = (0.2 * network.node[0].energy_max - network.node[0].energy_thresh) / network.node[0].energy_max

    step = (high_bound - low_bound) / 3

    queue_length = len(network.mc.list_request)
    min_energy = network.node[network.find_min_node()].energy

    idx = 0
    if queue_length > 0 and queue_length <= 3:
        if min_energy > 2 * network.node[0].energy_thresh / 3:
            idx = 3
        elif min_energy > 1 * network.node[0].energy_thresh / 3 and min_energy <= 2 * network.node[0].energy_thresh / 3:
            idx = 2
        elif min_energy > 0 and min_energy <= 1 * network.node[0].energy_thresh / 3:
            idx = 1
        else: 
            idx = 0
    elif queue_length > 3 and queue_length <= 10:
        if min_energy > 2 * network.node[0].energy_thresh / 3:
            idx = 2
        elif min_energy > 1 * network.node[0].energy_thresh / 3 and min_energy <= 2 * network.node[0].energy_thresh / 3:
            idx = 1
        elif min_energy > 0 and min_energy <= 1 * network.node[0].energy_thresh / 3:
            idx = 0
        else: 
            idx = 0
    elif queue_length > 10:
        if min_energy > 2 * network.node[0].energy_thresh / 3:
            idx = 1
        elif min_energy > 1 * network.node[0].energy_thresh / 3 and min_energy <= 2 * network.node[0].energy_thresh / 3:
            idx = 0
        elif min_energy > 0 and min_energy <= 1 * network.node[0].energy_thresh / 3:
            idx = 0
        else: 
            idx = 0

    theta = low_bound + idx * step
    return theta

def reward_function(network, q_learning, state, receive_func=find_receiver):
    """
    calculate each part of reward
    :param network:
    :param q_learning:
    :param state:
    :param receive_func:
    :return: each part of reward and charging time when mc stand at state
    """
    d = [distance.euclidean(item.location, q_learning.action_list[state]) for item in
         network.node]
    p = [para.alpha / (item + para.beta) ** 2 for item in d]
    index_negative = [index for index, node in enumerate(network.node) if p[index] < node.avg_energy]
    E = np.asarray([network.node[index].energy for index in index_negative])
    pe = np.asarray([p[index] / network.node[index].avg_energy for index in index_negative])
    if len(index_negative):
        min_E = min(E)
        min_pe = min(pe)
    else:
        min_E = min([node.energy for node in network.node])
        min_pe = 1.0
    # alpha = Fuzzy_Fix.get_output(min_E, len(index_negative), min_pe)
    alpha = determine_theta_heuristic(network)

    print(f'THETA: {alpha}')
    # if alpha > ((0.2 * network.node[0].energy_max - network.node[0].energy_thresh) / network.node[0].energy_max):
    #     # print(f'energy max: {network.node[0].energy_max}, energy thres: {network.node[0].energy_thresh}')
    #     alpha = (0.2 * network.node[0].energy_max - network.node[0].energy_thresh) / network.node[0].energy_max
    # elif alpha < ((2 * network.node[0].energy_thresh - network.node[0].energy_thresh) / network.node[0].energy_max):
    #     alpha = (2 * network.node[0].energy_thresh - network.node[0].energy_thresh) / network.node[0].energy_max

    charging_time = get_charging_time(network, q_learning, state, alpha)
    w, nb_target_alive = get_weight(network, network.mc, q_learning, state, charging_time, receive_func)
    p = get_charge_per_sec(network, q_learning, state)
    p_hat = p / np.sum(p)
    E = np.asarray([network.node[request["id"]].energy for request in network.mc.list_request])
    e = np.asarray([request["avg_energy"] for request in network.mc.list_request])
    second = nb_target_alive / len(network.target)
    third = np.sum(w * p_hat)
    first = np.sum(e * p / E)
    return first, second, third, charging_time


def init_function(nb_action=81):
    """
    init q table
    :param nb_action:
    :return:
    """
    return np.zeros((nb_action + 1, nb_action + 1), dtype=float)


def action_function(nb_action=225):
    """
    init action
    :param nb_action:
    :return:
    """
    list_action = []
    for i in range(int(math.sqrt(nb_action))):
        for j in range(int(math.sqrt(nb_action))):
            list_action.append((13 * (i + 1), 13 * (j + 1)))
    list_action.append(para.depot)
    return list_action


def action_function_modify(list_node):
    """
    init action
    :param nb_action:
    :return:
    """
    list_action = []
    for node in list_node:
        list_action.append(node.location)

    list_action.append(para.depot)
    # print(f'list action: {list_action}')
    return list_action

# def action_function(network):
#     list_action = []
#     for node in network.node:
#         list_action.append(node.location)
#     list_action.append(para.depot)
#     return list_action


def get_weight(net, mc, q_learning, action_id, charging_time, receive_func=find_receiver):
    """
    getting weight of each sensor. weight depends on the number of path which include the node
    :param net:
    :param mc:
    :param q_learning:
    :param action_id:
    :param charging_time:
    :param receive_func:
    :return: weight and the number of target is alive
    """
    p = get_charge_per_sec(net, q_learning, action_id)
    all_path = get_all_path(net, receive_func)
    time_move = distance.euclidean(q_learning.action_list[q_learning.state],
                                   q_learning.action_list[action_id]) / mc.velocity
    list_dead = []
    w = [0 for _ in mc.list_request]
    for request_id, request in enumerate(mc.list_request):
        temp = (net.node[request["id"]].energy - time_move * request["avg_energy"]) + (
            p[request_id] - request["avg_energy"]) * charging_time
        if temp < 0:
            list_dead.append(request["id"])
    for request_id, request in enumerate(mc.list_request):
        nb_path = 0
        for path in all_path:
            if request["id"] in path:
                nb_path += 1
        w[request_id] = nb_path
    total_weight = sum(w) + len(w) * 10 ** -3
    w = np.asarray([(item + 10 ** -3) / total_weight for item in w])
    nb_target_alive = 0
    for path in all_path:
        if para.base in path and not (set(list_dead) & set(path)):
            nb_target_alive += 1
    return w, nb_target_alive


def get_path(net, sensor_id, receive_func=find_receiver):
    """
    getting path from sensor_id to base
    :param net:
    :param sensor_id:
    :param receive_func:
    :return:
    """
    path = [sensor_id]
    if distance.euclidean(net.node[sensor_id].location, para.base) <= net.node[sensor_id].com_ran:
        path.append(para.base)
    else:
        receive_id = receive_func(net=net, node=net.node[sensor_id])
        if receive_id != -1:
            path.extend(get_path(net, receive_id, receive_func))
    return path


def get_all_path(net, receive_func=find_receiver):
    """
    getting all paths from every target to base
    :param net:
    :param receive_func:
    :return:
    """
    list_path = []
    for sensor_id, target_id in enumerate(net.target):
        list_path.append(get_path(net, sensor_id, receive_func))
    return list_path


def get_charge_per_sec(net, q_learning, state):
    """
    estimate energy which mc charge for node when standing at state
    :param net:
    :param q_learning:
    :param state:
    :return:
    """
    return np.asarray(
        [para.alpha / (distance.euclidean(net.node[request["id"]].location,
                                          q_learning.action_list[state]) + para.beta) ** 2 for
         request in net.mc.list_request])


# def get_charging_time(network=None, q_learning=None, state=None, alpha=0, charge_per_sec=get_charge_per_sec):
#     if not len(network.mc.list_request):
#         return 0
#
#     model = LpProblem("Find optimal time", LpMaximize)
#     T = LpVariable("Charging time", lowBound=0, upBound=None, cat=LpContinuous)
#     a = LpVariable.matrix("a", list(range(len(network.mc.list_request))), lowBound=0, upBound=1, cat="integer")
#     p = charge_per_sec(network, q_learning, state)
#     count = 0
#
#     for index, request in enumerate(network.mc.list_request):
#         if p[index] - request["avg_energy"] > 0:
#             print("charging time =", p[index] - request["avg_energy"])
#             count += 1
#             model += network.node[request["id"]].energy - distance.euclidean(q_learning.action_list[q_learning.state],
#                                                                              q_learning.action_list[
#                                                                                  state]) / network.mc.velocity * \
#                      request[
#                          "avg_energy"] + (
#                              p[index] - request["avg_energy"]) * T >= network.node[
#                          request["id"]].energy_thresh + alpha * network.node[request["id"]].energy_max - 10 ** 5 * (
#                              1 - a[index])
#             model += network.node[request["id"]].energy - distance.euclidean(q_learning.action_list[q_learning.state],
#                                                                              q_learning.action_list[
#                                                                                  state]) / network.mc.velocity * \
#                      request[
#                          "avg_energy"] + (
#                              p[index] - request["avg_energy"]) * T <= network.node[
#                          request["id"]].energy_thresh + alpha * network.node[request["id"]].energy_max + 10 ** 5 * a[
#                          index]
#             model += network.node[request["id"]].energy - distance.euclidean(q_learning.action_list[q_learning.state],
#                                                                              q_learning.action_list[
#                                                                                  state]) / network.mc.velocity * \
#                      request[
#                          "avg_energy"] + (
#                              p[index] - request["avg_energy"]) * T <= network.node[request["id"]].energy_max
#     print("count =", count)
#     if not count:
#         model += T == min(
#             [(- network.node[request["id"]].energy + distance.euclidean(q_learning.action_list[q_learning.state],
#                                                                         q_learning.action_list[
#                                                                             state]) / network.mc.velocity * request[
#                   "avg_energy"] + network.node[request["id"]].energy_max) / (- p[index] + request["avg_energy"])
#              for index, request in enumerate(network.mc.list_request)])
#     print(model.constraints)
#     model += lpSum(a)
#     status = model.solve()
#     print("status =", q_learning.action_list[state], value(T))
#     return value(T)


def get_charging_time(network=None, q_learning=None, state=None, alpha=0, is_test=False):
    """
    get charging time when mc stand at the state
    :param network:
    :param q_learning:
    :param state:
    :param alpha:
    :return:
    """
    # request_id = [request["id"] for request in network.mc.list_request]
    time_move = distance.euclidean(network.mc.current, q_learning.action_list[state]) / network.mc.velocity
    energy_min = network.node[0].energy_thresh + alpha * network.node[0].energy_max

    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    ds = []
    max_dist = 75.0
    for node in network.node:
        d = distance.euclidean(q_learning.action_list[state], node.location)
        ds.append(d)
        p = para.alpha / (d + para.beta) ** 2

        # if state == 73:
        #     print(f'energy min: {energy_min}, node energy: {node.energy}, energy loss moving: {time_move * node.avg_energy}, p: {p}, consum energy: {node.avg_energy}')
        if node.energy - time_move * node.avg_energy < energy_min and p - node.avg_energy > 0:
            s1.append((node.id, p))
            continue

        energy_min_s2 = node.energy_thresh + (node.avg_energy - p) * max_dist / network.mc.velocity
        if node.energy - time_move * node.avg_energy > energy_min_s2 and p - node.avg_energy < 0:
            s2.append((node.id, p, energy_min_s2))
    
    if len(s1) == 0:
        # print(f's1: {s1}')
        # print(f's2: {s2}')
        return 0

    t = []
    if state == 197:
        print(f'd: {ds}')
        print(f's1: {s1}')
        print(f's2: {s2}')
    
    for index, p in s1:
        t.append((energy_min - network.node[index].energy + time_move * network.node[index].avg_energy) / (
            p - network.node[index].avg_energy))
    for index, p, energy_min_s2 in s2:
        if (network.node[index].energy  - time_move * node.avg_energy > energy_min):
            t.append((energy_min - network.node[index].energy + time_move * network.node[index].avg_energy) / (
            p - network.node[index].avg_energy))
        else:
            t.append((energy_min_s2 - network.node[index].energy + time_move * network.node[index].avg_energy) / (
            p - network.node[index].avg_energy))
    
    if state == 197:
        print(f't: {t}')

    dead_list = []
    for item in t:
        nb_dead = 0
        # for index, node in enumerate(network.node):
        #     p = para.alpha / (distance.euclidean(node.location, q_learning.action_list[state])+para.beta)**2
        #     temp = node.energy - time_move * node.avg_energy + (p-node.avg_energy)*item
        #     if temp < energy_min:
        #         nb_dead += 1
        for index, p in s1:
            temp = network.node[index].energy - time_move * network.node[index].avg_energy + (
                p - network.node[index].avg_energy) * item
            if temp >= energy_min:
                nb_dead -= 1
        for index, p, energy_min_s2 in s2:
            temp = network.node[index].energy - time_move * network.node[index].avg_energy + (
                p - network.node[index].avg_energy) * item
            if (network.node[index].energy  - time_move * node.avg_energy > energy_min):
                if temp <= energy_min:
                    nb_dead += 1
            else:
                if temp <= energy_min_s2:
                    nb_dead += 1
        dead_list.append(nb_dead)
    
    if state == 197:
        print(f'dead node: {dead_list}')
    
    arg_min = np.argmin(dead_list)
    min_time = [t[index] for index, item in enumerate(dead_list) if dead_list[arg_min] == item]
    # print("t = ", t)
    # print("dead list = ", dead_list)
    # return t[arg_min]
    return min(min_time)
