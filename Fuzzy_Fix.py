import csv
import operator
import Parameter as param

E_max = param.e_max_sensor
def min_E(avg):
    membership = {"l": 0, "m": 0, "h": 0}
    if avg < 0.3 * E_max:
        membership["l"] = 1
    elif avg < 0.5 * E_max:
        membership["l"] = (0.5 * E_max - avg) / (0.2 * E_max)

    if 0.25 >= avg > 0.15:
        membership["m"] = 10 * (avg - 0.15)
    elif 0.35 >= avg > 0.25:
        membership["m"] = 10 * (0.35 - avg)

    if avg > 0.7 * E_max:
        membership["h"] = 1
    elif 0.5 <= avg <= 0.7:
        membership["h"] = (avg - 0.5 * E_max) / (0.2 * E_max)

    return membership


def p_e(pe):
    membership = {"l": 0, "m": 0, "h": 0}
	# """
    # if pe < 0.1:
    #     membership["l"] = 1
    # elif pe < 0.3:
    #     membership["l"] = 5 * (0.3 - pe)

    # if pe > 0.5:
    #     membership["h"] = 1
    # elif pe > 0.3:
    #     membership["h"] = 5 * (pe - 0.3)

    # if 0.1 <= pe < 0.3:
    #     membership["m"] = 5 * (pe - 0.1)
    # elif 0.3 <= pe <= 0.5:
    #     membership["m"] = 5 * (0.5 - pe)
	# """
    membership["l"] = 0
    membership["m"] = 1
    membership["h"] = 0

    return membership


def len_E(std):
    membership = {"l": 0, "m": 0, "h": 0}
    if std <= 1:
        membership["l"] = 1
    elif std <= 3:
        membership["l"] = 0.5 * (3 - std)

    if std >= 6:
        membership["h"] = 1
    elif std >= 4:
        membership["h"] = 0.5 * (std - 4)

    if 1 <= std <= 3:
        membership["m"] = 0.5 * (std - 1)
    elif 3 <= std <= 4:
        membership["m"] = 1
    elif 4 <= std <= 6:
        membership["m"] = 0.5 * (6 - std)

    return membership


def get_value(str1):
    if str1 == "l":
        out = -1
    elif str1 == "m":
        out = 0
    else:
        out = 1
    return out


def rule(avg, std, pe):
    out = get_value(avg) - get_value(std) + get_value(pe)
    if out == -3 or out == -2:
        temp = "vl"
    elif out == -1:
        temp = "l"
    elif out == 0:
        temp = "m"
    elif out == 1:
        temp = "h"
    else:
        temp = "vh"
    return temp


def get_output(avg, std, pe):
    temp_avg = min_E(avg)
    temp_len = len_E(std)
    temp_pe = p_e(pe)
    temp = dict()
    for key_avg, value_avg in temp_avg.items():
        for key_std, value_std in temp_len.items():
            for key_pe, value_pe in temp_pe.items():
                out = rule(key_avg, key_std, key_pe)
                out_value = min(value_avg, value_std, value_pe)
                temp[out] = max(temp.get(out, -1), out_value)
    r = max(temp.items(), key=operator.itemgetter(1))[0]
    if r == "vl":
        output = -0.1
    elif r == "l":
        output = 0.0
    elif r == "m":
        output = 0.1
    elif r == "h":
        output = 0.2
    elif r == "vh":
        output = 0.3
    return output