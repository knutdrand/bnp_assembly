from typing import Iterable, Dict, Union

from bnp_assembly.distance_matrix import DirectedDistanceMatrix
from bnp_assembly.graph_objects import Edge
from bnp_assembly.io import PairedReadStream
from bnp_assembly.location import LocationPair
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix


class EdgeDistanceFinder:
    """
    Interface for finding distance for edges (distance between two node sides)
    """
    def __call__(self, reads: Union[PairedReadStream, SparseInteractionMatrix], effective_contig_sizes) -> DirectedDistanceMatrix:  #Dict[Edge, float]:
        pass


