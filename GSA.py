import math
import numpy as np
from scipy.spatial import distance

import Parameter as para
import random


class GSA:
    def __init__(self):
        self.scheme = []
        self.nb_agent = 60
        self.t_max = 200
        self.g_0 = 100
        self.beta = 20
        self.epsilon = 10 ** -3

    def update(self, network):
        request_list = network.mc.list_request
        if not len(request_list):
            return network.mc.current, 0
        if len(request_list) == 1:
            p = para.alpha / para.beta ** 2
            id_node = network.mc.list_request[0]["id"]
            c_time = (network.node[id_node].energy_max - network.node[id_node].energy) / p
            return network.node[id_node].location, c_time
        if not self.scheme:
            self.gsa(network)
        id_node = self.scheme[0]
        p = para.alpha / para.beta ** 2
        c_time = (network.node[id_node].energy_max - network.node[id_node].energy) / p
        del self.scheme[0]
        print("location = ", network.node[id_node].location, "charge time = ", c_time)
        return network.node[id_node].location, c_time

    def gsa(self, network):
        population = self.population(network)
        t = -1
        while t < self.t_max:
            t = t + 1
            best, worst, id_best = self.best_worst(population)
            m = self.m(population, best, worst)
            for i, _ in enumerate(population):
                f_i = []
                for j, _ in enumerate(population):
                    if j == i:
                        continue
                    f_j = self.g(t) * m[i] * m[j] / (
                            distance.euclidean(population[i]["scheme"], population[j]["scheme"]) + self.epsilon) * (
                                  population[i]["scheme"] - population[j]["scheme"])
                    f_i.append(random.random() * f_j)
                #  update f_i
                f_i = np.asarray(f_i)
                f_i = sum(f_i)
                #  update a_i
                a_i = f_i / m[i]
                #  update velocity
                population[i]["velocity"] = np.random.rand(len(a_i)) * population[i]["velocity"] + a_i
                #  update scheme
                population[i]["scheme"] = population[i]["scheme"] + population[i]["velocity"]
                population[i]["fitness"] = self.fitness(network, population[i]["scheme"])
        best, worst, id_best = self.best_worst(population)
        scheme = population[id_best]["scheme"]
        #  get scheme id node
        list_request = network.mc.list_request
        # print("list_request = ", list_request)
        temp_scheme = [{"id": list_request[i]["id"], "value": scheme[i]} for i in range(len(list_request))]
        temp_scheme = sorted(temp_scheme, key=lambda i: i['value'])
        self.scheme = [item["id"] for item in temp_scheme]

    def population(self, network):
        pop = []
        # print(network.mc.list_request)
        for i in range(self.nb_agent):
            scheme = 2 * np.random.rand(len(network.mc.list_request)) - 1
            velocity = 0.0 * np.random.rand(len(network.mc.list_request))
            fitness = self.fitness(network, scheme)
            pop.append({"scheme": scheme, "velocity": velocity, "fitness": fitness})
        # for i in pop:
        #     print(i)
        return pop

    def fitness(self, network, scheme):
        p = para.alpha / para.beta ** 2
        list_request = network.mc.list_request
        temp_scheme = [{"id": list_request[i]["id"], "value": scheme[i]} for i in range(len(list_request))]
        temp_scheme = sorted(temp_scheme, key=lambda i: i['value'])
        mc_current = network.mc.current
        latency = []
        w_time = 0
        for i in range(len(scheme)):
            id_node = temp_scheme[i]["id"]
            t_time = distance.euclidean(mc_current, network.node[id_node].location) / network.mc.velocity
            c_time = (network.node[id_node].energy_max - network.node[id_node].energy) / p
            w_time = w_time + t_time + c_time
            latency.append(w_time)
            #  update location of mc after charging
            mc_current = network.node[id_node].location
        return sum(latency) / len(latency)

    def g(self, t):
        return self.g_0 * math.exp(-self.beta * t / self.t_max)

    def m(self, population, best, worst):
        m = []
        for i, individual in enumerate(population):
            m.append((individual["fitness"] - worst + self.epsilon) / (best - worst + self.epsilon))
        return np.array([item / sum(m) for item in m])

    def best_worst(self, population):
        id_best = -1
        best = float("inf")
        worst = float("-inf")
        for id_individual, individual in enumerate(population):
            if individual["fitness"] < best:
                id_best = id_individual
                best = individual["fitness"]
            if individual["fitness"] > worst:
                worst = individual["fitness"]
        return best, worst, id_best
