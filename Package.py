import Parameter as para


class Package:
    def __init__(self, is_energy_info=False):
        self.path = []  # the path include nodes in path from node to base
        self.is_energy_info = is_energy_info  # if package is energy package, is_energy_info = True
        self.size = self.get_size()
        self.is_success = False  # is_success = True if package can go to base

    def get_size(self):
        """
        :return: size of package
        """
        return para.b if not self.is_energy_info else para.b_energy

    def update_path(self, node_id):
        """
        :param node_id: the id of node in the path of package
        :return: update the path of package
        """
        self.path.append(node_id)
