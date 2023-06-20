from bnp_assembly.location import LocationPair


class ScaffolderInterface:
    def __init__(self, contig_dict):
        self._contig_dict = contig_dict

    def preprocess(self):
        return NotImplemented

    def register_read_pairs(self, read_pairs: LocationPair):
        return NotImplemented

    def get_distance_matrix(self):
        return NotImplemented

    def get_scaffold(self, location_pairs):
        self.register_read_pairs(location_pairs)
        distance_matrix = self.get_distance_matrix()
        return self.get_scaffold_from_distance_matrix(distance_matrix)

    def get_scaffold_from_distance_matrix(self, distance_matrix):
        pass

