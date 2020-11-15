from Node import Node
import random
from Network import Network
import pandas as pd
from ast import literal_eval
from MobileCharger import MobileCharger
from Q__Learning import Q_learning
from Inma import Inma
from GSA import GSA
from scipy.stats import sem, t
import numpy as np
import csv
import sys

# index = 0
# random.seed(3)
# df = pd.read_csv("data/thaydoitileguitin.csv")
# node_pos = list(literal_eval(df.node_pos[index]))
# list_node = []
# for i in range(len(node_pos)):
#     location = node_pos[i]
#     com_ran = df.commRange[index]
#     energy = df.energy[index]
#     energy_max = df.energy[index]
#     prob = df.freq[index]
#     node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
#                 energy_thresh=0.4 * energy, prob=prob)
#     list_node.append(node)
# mc = MobileCharger(energy=df.E_mc[index], capacity=df.E_max[index], e_move=df.e_move[index],
#                    e_self_charge=df.e_mc[index], velocity=df.velocity[index])
# target = [int(item) for item in df.target[index].split(',')]
# net = Network(list_node=list_node,  mc=mc, target=target)
# print(len(net.node), len(net.target), max(net.target))
# q_learning = Q_learning(network=net)
# inma = Inma()
# gsa = GSA()
# net.simulate(optimizer=q_learning, file_name="log/energy_information_log.csv")


read_file = sys.argv[1]
write_file = sys.argv[2]
data_range = int(sys.argv[3])
data_start = int(sys.argv[4])
run_range = int(sys.argv[5])
learning_rate = float(sys.argv[6])
scale_factor = float(sys.argv[7])
read_name = "data/" + read_file + ".csv"
try:
    max_time = int(sys.argv[8])
except:
    max_time = None
df = pd.read_csv(read_name)
for id_data in range(data_range):
    index = id_data + data_start
    print("nb data = ", index)
    write_name = "log/" + write_file + str(index) + ".csv"
    open_file = open(write_name, "w")
    result = csv.DictWriter(open_file, fieldnames=["nb run", "lifetime"])
    result.writeheader()
    life_time = []
    for nb_run in range(run_range):
        print("nb run = ", nb_run)
        random.seed(nb_run)
        node_pos = list(literal_eval(df.node_pos[index]))
        list_node = []
        for i in range(len(node_pos)):
            location = node_pos[i]
            com_ran = df.commRange[index]
            energy = df.energy[index]
            energy_max = df.energy[index]
            prob = df.freq[index]
            node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                        energy_thresh=0.4 * energy, prob=prob)
            list_node.append(node)
        mc = MobileCharger(energy=df.E_mc[index], capacity=df.E_max[index], e_move=df.e_move[index],
                           e_self_charge=df.e_mc[index], velocity=df.velocity[index])
        target = [int(item) for item in df.target[index].split(',')]
        net = Network(list_node=list_node, mc=mc, target=target)
        # print(len(net.node), len(net.target), max(net.target))
        q_learning = Q_learning(alpha=learning_rate, gamma=scale_factor)
        inma = Inma()
        gsa = GSA()
        file_name = "log/q_learning_" + str(index) + ".csv"
        temp = net.simulate(optimizer=q_learning, file_name=file_name, max_time=max_time)
        life_time.append(temp)
        result.writerow({"nb run": nb_run, "lifetime": temp})
        print("done run = ", nb_run)

    confidence = 0.95
    h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
    result.writerow({"nb run": np.mean(life_time), "lifetime": h})
    open_file.close()
    print("done data = ", index)
print("done all")
