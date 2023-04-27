from dataclasses import dataclass
from bionumpy.bnpdataclass import bnpdataclass
import numpy as np
from typing import Dict
from ..graph_objects import NodeSide
from ..contig_graph import ContigPath
@dataclass
class SimulationParams:
    n_nodes: int
    n_reads: int
    node_length: int = 100



class PairDistribution:
    def __init__(self, contig_length, p):
        self._contig_length = contig_length
        self._p = p

    def sample(self, rng, n_samples=1):
        distance = np.minimum(rng.geometric(self._p, size=n_samples), self._contig_length)
        first = rng.integers(0, self._contig_length-distance)
        return (first, first+distance)

@bnpdataclass
class Location:
    contig_id: int
    offset: int


@dataclass
class ContigSplit:
    contig_length: int
    starts: np.ndarray

    def get_contig_dict(self) -> Dict[int, int]:
        lengths = np.append(self.starts, self.contig_length)[1:]-self.starts
        return dict(enumerate(lengths))

    def get_paths(self):
        node_sides = []
        for i in range(len(self.starts)):
            node_sides.append(NodeSide(i, 'l'))
            node_sides.append(NodeSide(i, 'r'))
        return [ContigPath.from_node_sides(node_sides)]

    def map(self, positions):
        contig_ids = np.searchsorted(self.starts, positions,side='right')-1
        offsets = positions-self.starts[contig_ids]
        return Location(contig_ids, offsets)

def split_contig(rng, contig_length, n_parts):
    split_points = rng.choice(contig_length-1, size=n_parts-1, replace=False)+1
    return ContigSplit(contig_length, np.insert(np.sort(split_points), 0, 0))

def split_contig_on_size(contig_length, size):
    split_points = np.arange(0, contig_length, size)
    return ContigSplit(contig_length, split_points)


@dataclass
class SplitAndPairs:
    split: ContigSplit
    location_a: Location
    location_b: Location

def simulate_split_contig_reads(contig_length, n_parts, n_pairs, p=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    first, second = PairDistribution(contig_length, 0.1).sample(rng, n_pairs)
    contig_split = split_contig_on_size(contig_length, contig_length//n_parts)
    first_split = contig_split.map(first)
    second_split = contig_split.map(second)
    return SplitAndPairs(contig_split, first_split, second_split)


def simulate_merged_contig_reads(node_length, n_parts, n_pairs, p=0.1, rng=None):
    contig_length = node_length*n_parts
    if rng is None:
        rng = np.random.default_rng()
    first, second = PairDistribution(contig_length, 0.1).sample(rng, n_pairs)
    contig_split = split_contig_on_size(contig_length, contig_length//n_parts)
    first_split = contig_split.map(first)
    second_split = contig_split.map(second)
    return SplitAndPairs(contig_split, first_split, second_split)


def test():
    return simulate_split_contig_reads(100, 5, 10)
