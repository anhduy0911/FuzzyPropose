import csv


def get_type(x):
    if x == -1:
        return "L"
    elif x == 0:
        return "M"
    else:
        return "H"


def get_out(x):
    if x == -3:
        return "VVL"
    if x <= -2:
        return "VL"
    elif x == -1:
        return "L"
    elif x == 0:
        return "M"
    elif x == 1:
        return "H"
    elif x == 2:
        return "VH"
    else:
        return "VVH"


f = open("fuzzy/rule.csv", "w")
writer = csv.DictWriter(f, fieldnames=["sum_E", "sigma_E", "p_e", "theta"])
writer.writeheader()
for i in range(3):
    sum_E = i - 1
    theta1 = sum_E
    for j in range(3):
        sigma_E = j - 1
        theta2 = - sigma_E
        for k in range(3):
            p_e = k - 1
            theta3 = p_e
            theta = theta1 + theta2 + theta3
            row = {"sum_E": get_type(sum_E), "sigma_E": get_type(sigma_E), "p_e": get_type(p_e),
                   "theta": get_out(theta)}
            writer.writerow(row)
f.close()
