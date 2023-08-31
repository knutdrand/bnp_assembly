import typing as tp
from abc import abstractmethod

from bnp_assembly.location import LocationPair


class EdgeScorer:
    @abstractmethod
    def __init__(self, contig_dict: tp.Dict[int, int], read_pairs: tp.List[LocationPair]):
        pass

    @abstractmethod
    def get_distance_matrix(self):
        pass

