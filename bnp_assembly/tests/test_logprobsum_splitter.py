import dataclasses

import numpy as np
import plotly.express as px
from numpy.testing import assert_array_equal

from bnp_assembly.contig_graph import DirectedNode, ContigPath
from bnp_assembly.distance_distribution import DistanceDistribution
from bnp_assembly.location import Location, LocationPair
from bnp_assembly.logprobsum_splitter import LogsumprobMatrices, BaseProbabilityMatrices

contig_path = ContigPath.from_directed_nodes([
    DirectedNode(i, '+') for i in range(3)])

size_array = np.array([10, 20, 30])

distance_distribution = DistanceDistribution(np.log(np.arange(1, 20)[::-1] / np.sum(np.arange(1, 20))))
location_pairs = LocationPair(
    Location([0, 1, 2], [0, 1, 2]),
    Location([2, 1, 0], [0, 1, 2]))

location_pairs_iter = [LocationPair(
    Location(np.full(100, 1), np.full(100, 19)),
    Location(np.full(100, 2), np.full(100, 1))),
    LocationPair(Location(np.tile([0, 1, 2], 20), np.full(60, 5)),
                 Location(np.repeat([0, 1, 2], 20), np.full(60, 5)))]


@dataclasses.dataclass
class InputData:
    size_array: np.array
    contig_path: ContigPath
    distance_distribution: DistanceDistribution

simple_input = InputData(np.array([5]),
                         ContigPath.from_directed_nodes([DirectedNode(0, '+')]),
                         DistanceDistribution(np.full(3, np.log(1/3))))
uniform_distribution = lambda x: DistanceDistribution(np.full(x, np.log(1/x)))
simple_path = lambda n_nodes: ContigPath.from_directed_nodes([DirectedNode(i, '+') for i in range(n_nodes)])
two_input = InputData(np.array([3, 2]), simple_path(2), uniform_distribution(3))
def test_base_probability():
    '''
    Dxx00
    xDxx0
    xxDxx
    0xxDx
    00xxD
    '''

    base_probs = BaseProbabilityMatrices(simple_input.size_array, simple_input.contig_path, simple_input.distance_distribution)
    assert_array_equal(base_probs.A, [[25-6]])

def test_two():
    base_probs = BaseProbabilityMatrices(two_input.size_array, two_input.contig_path, two_input.distance_distribution)
    assert np.sum(base_probs.A) == 19


def test_prob():
    base_probs = BaseProbabilityMatrices(two_input.size_array, two_input.contig_path, DistanceDistribution(np.array([np.log(9/10), np.log(1/10)])))
    np.testing.assert_allclose(np.sum(base_probs.N), 5*9/10+2*4*1/10)

def test_logsumprob_acceptance():
    logsumprobmat = LogsumprobMatrices(size_array, contig_path, distance_distribution)
    logsumprobmat.register_location_pairs(location_pairs)


def test_reasonable():
    logsumprobmat = LogsumprobMatrices(size_array, contig_path, distance_distribution)
    for location_pairs in location_pairs_iter:
        logsumprobmat.register_location_pairs(location_pairs)
    connected, disconnected = logsumprobmat.matrices
    # px.imshow(connected, title='connected').show()
    # px.imshow(disconnected, title='disconnected').show()
