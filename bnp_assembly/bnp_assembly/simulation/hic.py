from dataclasses import dataclass, replace
from bionumpy.bnpdataclass import bnpdataclass
import numpy as np
from typing import Dict, List

from .pair_distribution import PairDistribution
from ..graph_objects import NodeSide
from ..contig_graph import ContigPath, DirectedNode


@dataclass
class SimulationParams:
    n_nodes: int
    n_reads: int
    node_length: int = 100
    mean_distance: int = 10
    n_chromosomes: int = 1
    noise_proportion: float = 0.0


@bnpdataclass
class Location:
    contig_id: int
    offset: int


@dataclass
class ContigSplit:
    contig_length: int
    starts: np.ndarray

    def get_contig_dict(self) -> Dict[int, int]:
        lengths = np.append(self.starts, self.contig_length)[1:] - self.starts
        return dict(enumerate(lengths))

    def get_paths(self):
        node_sides = []
        for i in range(len(self.starts)):
            node_sides.append(NodeSide(i, 'l'))
            node_sides.append(NodeSide(i, 'r'))
        return [ContigPath.from_node_sides(node_sides)]

    def map(self, positions):
        contig_ids = np.searchsorted(self.starts, positions, side='right') - 1
        offsets = positions - self.starts[contig_ids]
        d = self.get_contig_dict()
        ls = [d[contig_id] for contig_id in contig_ids]
        assert np.all(offsets < ls), (offsets, ls)
        return Location(contig_ids, offsets)


@dataclass
class ManyContigSplit:
    contig_splits: List[ContigSplit]

    def get_contig_dict(self):
        offset = 0
        d = {}
        for contig_split in self.contig_splits:
            contig_dict = contig_split.get_contig_dict()
            d.update({key + offset: value for key, value in contig_dict.items()})
            offset += len(contig_dict)
        return d

    def get_paths(self):
        offset = 0
        paths = []
        for contig_split in self.contig_splits:
            path = contig_split.get_paths()[0]
            paths.append(ContigPath.from_directed_nodes(
                [DirectedNode(dn.node_id + offset, dn.orientation)
                 for dn in path.directed_nodes]))
            offset += len(contig_split.get_contig_dict())
        return paths

    @property
    def contig_offsets(self):
        return np.cumsum([0] + [len(contig_split.get_contig_dict()) for contig_split in self.contig_splits])[:-1]

    def map(self, contig_id, postions):
        location = self.contig_splits[contig_id].map(positions)
        return Location(self.contig_offsets[contig_id] + location.contig_id, location.offset)

    def map_locations(self, contig_id, locations):
        return Location(self.contig_offsets[contig_id] + locations.contig_id, locations.offset)


def split_contig(rng, contig_length, n_parts):
    split_points = rng.choice(contig_length - 1, size=n_parts - 1, replace=False) + 1
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
    contig_split = split_contig_on_size(contig_length, contig_length // n_parts)
    first_split = contig_split.map(first)
    second_split = contig_split.map(second)
    return SplitAndPairs(contig_split, first_split, second_split)


def simulate_merged_contig_reads(node_length, n_parts, n_pairs, p=0.1, rng=None):
    contig_length = node_length * n_parts
    if rng is None:
        rng = np.random.default_rng()
    print(',', p)
    first, second = PairDistribution(contig_length, p).sample(rng, n_pairs)
    contig_split = split_contig_on_size(contig_length, contig_length // n_parts)
    first_split = contig_split.map(first)
    second_split = contig_split.map(second)
    return SplitAndPairs(contig_split, first_split, second_split)


def simulate_many_contigs(n_contigs, *args, **kwargs):  # node_length, n_parts, n_pairs, p=0.1, rng=None):
    split_and_pair_list = [simulate_merged_contig_reads(*args, **kwargs) for i in range(n_contigs)]
    split = ManyContigSplit([sp.split for sp in split_and_pair_list])
    location_a = np.concatenate([split.map_locations(i, sp.location_a) for i, sp in enumerate(split_and_pair_list)])
    location_b = np.concatenate([split.map_locations(i, sp.location_b) for i, sp in enumerate(split_and_pair_list)])
    return SplitAndPairs(split, location_a, location_b)


def full_simulation(simulation_params: SimulationParams, rng: np.random.Generator = None):
    n_noise_reads = int(simulation_params.n_reads * simulation_params.noise_proportion)
    n_signal_reads = simulation_params.n_reads - n_noise_reads
    split_and_pairs = simulate_many_contigs(simulation_params.n_chromosomes, simulation_params.node_length,
                                            simulation_params.n_nodes, n_signal_reads,
                                            1 / simulation_params.mean_distance, rng=rng)
    noise_a, noise_b = simulate_noise(split_and_pairs, n_noise_reads, rng=rng)
    return replace(split_and_pairs,
                   location_a=np.concatenate([split_and_pairs.location_a, noise_a]),
                   location_b=np.concatenate([split_and_pairs.location_b, noise_b]))


def simulate_noise(split_and_pairs, n_noise, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    contig_split = split_and_pairs.split
    contig_dict = contig_split.get_contig_dict()
    total_size = sum(contig_dict.values())
    sizes = np.array(list(contig_dict.values()))
    ps = sizes / total_size
    contig_ids = list(contig_dict.keys())
    simulated_contig_ids = (rng.choice(contig_ids, p=ps, size=(2, n_noise), replace=True))
    offsets = rng.uniform(0, sizes[simulated_contig_ids])
    return tuple(Location(simulated_contig_ids[i], offsets[i]) for i in range(2))


def test():
    return simulate_split_contig_reads(100, 5, 10)
