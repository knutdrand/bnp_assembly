import numpy as np

from bnp_assembly.contig_graph import DirectedNode, ContigPath
from bnp_assembly.distance_distribution import DistanceDistribution
from bnp_assembly.location import Location, LocationPair
from bnp_assembly.logprobsum_splitter import LogsumprobMatrices

contig_path = ContigPath.from_directed_nodes([
    DirectedNode(i, '+') for i in range(3)])

size_array = np.array([10, 20, 30])
distance_distribution = DistanceDistribution(np.log(np.arange(1, 20)[::-1] / np.sum(np.arange(1, 20))))
location_pairs = LocationPair(
    Location([0, 1, 2], [0, 1, 2]),
    Location([2, 1, 0], [0, 1, 2]))

location_pairs = LocationPair(
    Location(np.full(100, 1)), np.full(100, 5))

def test_logsumprob_acceptance():
    logsumprobmat = LogsumprobMatrices(size_array, contig_path, distance_distribution)
    logsumprobmat.register_location_pairs(location_pairs)
