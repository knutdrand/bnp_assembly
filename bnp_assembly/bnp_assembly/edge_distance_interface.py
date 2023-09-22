from typing import Iterable
from bnp_assembly.location import LocationPair


class EdgeDistanceFinder:
    """
    Interface for finding distance for edges (distance between two node sides)
    """
    def __call__(self, reads: Iterable[LocationPair]):
        pass


