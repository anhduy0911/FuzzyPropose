import csv
import numpy as np

from scipy.spatial import distance

import Parameter as para
from Network_Method import uniform_com_func, to_string, count_package_function


class Network:
    def __init__(self, list_node=None, mc=None, target=None):
        self.node = list_node  # list of node in network
        self.set_neighbor()  # find neighbor of each node
        self.set_level()  # set the distance in graph from each node to base
        self.mc = mc  # mobile charger
        self.target = target  # list of target. each item is index of sensor where target is located

    def set_neighbor(self):
        """
        find the neighbor of every node
        :return:
        """
        for node in self.node:
            for other in self.node:
                if other.id != node.id and distance.euclidean(node.location, other.location) <= node.com_ran:
                    node.neighbor.append(other.id)

    def set_level(self):
        """
        find the distance in graph from every node to base
        :return:
        """
        queue = []
        for node in self.node:
            if distance.euclidean(node.location, para.base) < node.com_ran:
                node.level = 1
                queue.append(node.id)
        while queue:
            for neighbor_id in self.node[queue[0]].neighbor:
                if not self.node[neighbor_id].level:
                    self.node[neighbor_id].level = self.node[queue[0]].level + 1
                    queue.append(neighbor_id)
            queue.pop(0)

    def communicate(self, communicate_func=uniform_com_func):
        """
        communicate each time in network
        :param communicate_func: the function used to communicating
        :return:
        """
        return communicate_func(self)

    def run_per_second(self, t, optimizer=None):
        """
        simulate network per second
        :param t: current time
        :param optimizer: the optimizer used to calculate the next location of mc
        :return:
        """
        state = self.communicate()
        request_id = []
        for index, node in enumerate(self.node):
            if node.energy < node.energy_thresh:
                node.request(mc=self.mc, t=t)
                request_id.append(index)
            else:
                node.is_request = False
        if request_id:
            for index, node in enumerate(self.node):
                if index not in request_id and (t - node.check_point[-1]["time"]) > 50:
                    node.set_check_point(t)
        if optimizer:
            self.mc.run(network=self, time_stem=t, optimizer=optimizer)
        return state

    def simulate_lifetime(self, optimizer=None, file_name="log/energy_log.csv"):
        """
        simulate process finish when energy of any node is less than 0
        :param optimizer:
        :param file_name: log file
        :return:
        """
        energy_log = open(file_name, "w")
        writer = csv.DictWriter(energy_log, fieldnames=["time", "mc energy", "min energy"])
        writer.writeheader()
        t = 0
        while self.node[self.find_min_node()].energy >= 0 and t <= 2 * 10**6:
            t = t + 1
            if (t - 1) % 1000 == 0:
                print(t, self.mc.current, self.node[self.find_min_node()].energy)
            state = self.run_per_second(t, optimizer)
            # if not (t - 1) % 50:
            #     writer.writerow(
            #         {"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})
        writer.writerow({"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})
        energy_log.close()

    def simulate_max_time(self, optimizer=None, max_time=10**6, file_name="log/information_log.csv"):
        """
        simulate process finish when current time is more than the max_time
        :param optimizer:
        :param max_time:
        :param file_name:
        :return:
        """
        information_log = open(file_name, "w")
        writer = csv.DictWriter(information_log, fieldnames=["time", "nb dead", "nb package"])
        writer.writeheader()
        nb_dead = 0
        nb_package = len(self.target)
        t = 0
        while t <= max_time and nb_package > 0:
            t += 1
            if (t-1) % 1000 == 0:
                print(t, self.mc.current, self.node[self.find_min_node()].energy)
            state = self.run_per_second(t, optimizer)
            current_dead = self.count_dead_node()
            current_package = self.count_package()
            if current_dead != nb_dead or current_package != nb_package:
                nb_dead = current_dead
                nb_package = current_package
                writer.writerow({"time": t, "nb dead": nb_dead, "nb package": nb_package})
        information_log.close()

    def simulate(self, optimizer=None, max_time=None, file_name="log/energy_log.csv"):
        """
        simulate in general. if max_time is not none, simulate_max_time will be called
        :param optimizer:
        :param max_time:
        :param file_name:
        :return:
        """
        if max_time:
            self.simulate_max_time(optimizer=optimizer, max_time=max_time, file_name=file_name)
        else:
            self.simulate_lifetime(optimizer=optimizer, file_name=file_name)

    def print_net(self, func=to_string):
        """
        print information of network
        :param func:
        :return:
        """
        func(self)

    def find_min_node(self):
        """
        find id of node which has minimum energy in network
        :return:
        """
        min_energy = 10 ** 10
        min_id = -1
        for node in self.node:
            if node.energy < min_energy:
                min_energy = node.energy
                min_id = node.id
        return min_id

    def count_dead_node(self):
        """
        count the number of node which dead
        :return:
        """
        count = 0
        for node in self.node:
            if node.energy < 0:
                count += 1
        return count

    def count_package(self, count_func=count_package_function):
        """
        count the number of package which can go to base
        :param count_func:
        :return:
        """
        count = count_func(self)
        return count
