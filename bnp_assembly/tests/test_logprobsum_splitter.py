import numpy as np
import plotly.express as px
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

location_pairs_iter = [LocationPair(
    Location(np.full(100, 1), np.full(100, 19)),
    Location(np.full(100, 2), np.full(100, 1))),
    LocationPair(Location(np.tile([0,1,2],20), np.full(60, 5)),
                 Location(np.repeat([0,1,2],20), np.full(60, 5)))]


def test_logsumprob_acceptance():
    logsumprobmat = LogsumprobMatrices(size_array, contig_path, distance_distribution)
    logsumprobmat.register_location_pairs(location_pairs)

def test_reasonable():
    logsumprobmat = LogsumprobMatrices(size_array, contig_path, distance_distribution)
    for location_pairs in location_pairs_iter:
        logsumprobmat.register_location_pairs(location_pairs)
    connected, disconnected =  logsumprobmat.matrices
    px.imshow(connected, title='connected').show()
    px.imshow(disconnected, title='disconnected').show()
