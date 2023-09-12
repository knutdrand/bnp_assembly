from typing import Protocol, Callable
from ..simulations.hic import Location, SplitAndPairs
from ..contig_graph import ContigPath


class Scaffolding:
    def from_scaffold_alignments(self):
        pass


Scaffolder = Callable[[Location, Location], ContigPath]


def evaluate_scaffolder(scaffolder: Scaffolder, simulation_func: Callable[[], SplitAndPairs]):
    split_and_pairs = simulation_func()
    contig_path = scaffolder(split_and_pairs.location_a, split_and_pairs.location_b)

