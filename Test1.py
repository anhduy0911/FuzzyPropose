import random
import numpy as np
import matplotlib.pyplot as plt
import csv
np.random.seed(11)


# temp = []
# nb = 299
# for i in range(300):
#     r = random.randint(0, nb)
#     while r in temp:
#         r = random.randint(0, nb)
#     temp.append(r)
#
# print(temp)

# means = [[5.00, 5.00]]
# cov = [[1, 0], [0, 1]]
# N = 200
# X0 = np.random.multivariate_normal(means[0], cov, N) * 100
# X1 = np.random.multivariate_normal(means[1], cov, N) * 100
# X2 = np.random.multivariate_normal(means[2], cov, N) * 100
# X3 = np.random.multivariate_normal(means[3], cov, N) * 100
# X4 = np.random.multivariate_normal(means[4], cov, N) * 100
# X = np.concatenate((X0))
# f = open("log/cluster1.csv", "w")
# writer = csv.DictWriter(f, fieldnames=["target", "sensor"])
# for item in X0:
#     x = 1000 * random.random()
#     y = 1000 * random.random()
#     writer.writerow({"target": item, "sensor": np.array([x, y])})
# f.close()
#
# plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)
# # plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)
# # plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)
# # plt.plot(X3[:, 0], X3[:, 1], 'b^', markersize=4, alpha=.8)
# # plt.plot(X4[:, 0], X4[:, 1], 'go', markersize=4, alpha=.8)
# plt.xlim((0, 1000))
# plt.ylim((0, 1000))
# plt.show()

