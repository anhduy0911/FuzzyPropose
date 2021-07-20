import csv

import numpy as np
from scipy.spatial import distance

import Parameter as para
from Node_Method import find_receiver
from Q_learning_method import init_function, action_function, q_max_function, reward_function


class Q_learning:
    def __init__(self, init_func=init_function, nb_action=81, action_func=action_function, alpha=0.5, gamma=0.5):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # scale factor
        self.action_list = action_func(nb_action=nb_action)  # the list of action
        self.q_table = init_func(nb_action=nb_action)  # q table
        self.state = nb_action  # the current state of actor
        self.charging_time = [0.0 for _ in self.action_list]  # the list of charging time at each action
        self.reward = np.asarray([0.0 for _ in self.action_list])  # the reward of each action
        self.reward_max = [0.0 for _ in self.action_list]  # the maximum reward of each action

    def update(self, network, q_max_func=q_max_function, reward_func=reward_function):
        """
        update q table and all attribute of q learning
        :param network:
        :param alpha: learning rate
        :param gamma: learning rate
        :param q_max_func:
        :param reward_func:
        :return: next action and charging time at the location
        """
        if not len(network.mc.list_request):
            return self.action_list[self.state], 0.0
        self.set_reward(reward_func=reward_func, network=network)
        self.q_table[self.state] = (1 - self.alpha) * self.q_table[self.state] + self.alpha * (
            self.reward + self.gamma * self.q_max(q_max_func))
        self.choose_next_state(network)
        if self.state == len(self.action_list) - 1:
            charging_time = (network.mc.capacity - network.mc.energy) / network.mc.e_self_charge
        else:
            charging_time = self.charging_time[self.state]
        # if network.mc.list_request:
        #     d = [distance.euclidean(item.location, self.action_list[self.state]) for item in
        #          network.node]
        #     p = [para.alpha / (item + para.beta) ** 2 for item in d]
        #     index_negative = [index for index, node in enumerate(network.node) if p[index] < node.avg_energy]
        #     E = np.asarray([network.node[index].energy for index in index_negative])
        #     pe = np.asarray([p[index] / network.node[index].avg_energy for index in index_negative])
        #     # le = [index for index, node in enumerate(network.node) if
        #     #       p[index] < node.avg_energy and node.energy < 5]
        #     f = open("log/energy_info.csv", "a")
        #     writer = csv.DictWriter(f, fieldnames=["min E", "min pe", "len negative", "charge location", "charge time"])
        #     writer.writerow(
        #         {"min E": min(E), "min pe": min(pe), "len negative": len(index_negative),
        #          "charge location": self.action_list[self.state], "charge time": charging_time})
        #     f.close()
        return self.action_list[self.state], charging_time

    def q_max(self, q_max_func=q_max_function):
        """
        update q max
        :param q_max_func:
        :return:
        """
        return q_max_func(q_table=self.q_table, state=self.state)

    def set_reward(self, reward_func=reward_function, network=None):
        """
        update reward
        :param reward_func:
        :param network:
        :return:
        """
        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)
        third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index, row in enumerate(self.q_table):
            temp = reward_func(network=network, q_learning=self, state=index, receive_func=find_receiver)
            first[index] = temp[0]
            second[index] = temp[1]
            third[index] = temp[2]
            self.charging_time[index] = temp[3]
        first = first / np.sum(first)
        second = second / np.sum(second)
        third = third / np.sum(third)
        self.reward = first + second + third
        self.reward_max = list(zip(first, second, third))

    def choose_next_state(self, network):
        """
        choose next state of actor. next state is depot if mc is not enough energy
        :param network:
        :return:
        """
        # next_state = np.argmax(self.q_table[self.state])
        if network.mc.energy < 0.1:
            self.state = len(self.q_table) - 1
        else:
            self.state = np.argmax(self.q_table[self.state])
            # print(self.reward_max[self.state])
            # print(self.action_list[self.state])
