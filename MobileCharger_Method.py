from scipy.spatial import distance


def get_location(mc):
    """
    :param mc: mobile charger
    :return: the next location of mc
    """
    d = distance.euclidean(mc.start, mc.end)
    time_move = d / mc.velocity
    if time_move == 0:
        return mc.current
    elif distance.euclidean(mc.current, mc.end) < 10 ** -3:
        return mc.end
    else:
        x_hat = (mc.end[0] - mc.start[0]) / time_move + mc.current[0]
        y_hat = (mc.end[1] - mc.start[1]) / time_move + mc.current[1]
        if (mc.end[0] - mc.current[0]) * (mc.end[0] - x_hat) < 0 or (
                (mc.end[0] - mc.current[0]) * (mc.end[0] - x_hat) == 0 and (mc.end[1] - mc.current[1]) * (
                mc.end[1] - y_hat) <= 0):
            return mc.end
        else:
            return x_hat, y_hat


def charging(mc, net, node):
    """
    mc charge per second in network
    :param mc: mobile charger
    :param net: network
    :param node: the node, if multiple charging, node is none
    :return:
    """
    for nd in net.node:
        if distance.euclidean(nd.location, mc.current) < 1:
            p = nd.charge(mc)
            mc.energy -= p
