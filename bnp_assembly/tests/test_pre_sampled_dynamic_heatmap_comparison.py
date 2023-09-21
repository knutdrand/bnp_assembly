import numpy as np
from numpy.testing import assert_array_equal
import pytest
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmap, PreComputedDynamicHeatmapCreator, \
    DynamicHeatmapConfig
import bionumpy as bnp


from bnp_assembly.graph_objects import NodeSide, Edge
from bnp_assembly.location import Location, LocationPair
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmap, DynamicHeatmaps, HeatmapComparison

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])


def test_create_from_positions():
    dh = DynamicHeatmap.create_from_positions((a, b), n_bins=3, scale_func=lambda x: x)
    assert np.sum(dh.array) == 1
    assert_array_equal(dh.array, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
    dh = DynamicHeatmap.create_from_positions((a, b), n_bins=5, scale_func=lambda x: x)
    assert_array_equal(dh.array, np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0 ,0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0 ,0]]))


def test_add_sampled_read_pair_to_heatmap():
    c = PreComputedDynamicHeatmapCreator
    heatmap = DynamicHeatmap(np.zeros((10, 10)), lambda x: x)
    c.add_contig_offset_pair_to_heatmap(heatmap, 2, 6, gap=0, contig_size=10)
    assert np.sum(heatmap.array) == 1
    assert heatmap.array[2, 1] == 1
    c.add_contig_offset_pair_to_heatmap(heatmap, 0, 5, gap=0, contig_size=10)
    assert np.sum(heatmap.array) == 2
    assert heatmap.array[4, 0] == 1


@pytest.fixture
def config():
    return DynamicHeatmapConfig(scale_func=lambda x: x, inverse_scale_func=lambda x: x, n_bins=10)


@pytest.fixture
def genome():
    return {
        1: 10,
        2: 2,
        3: 20
    }


def test_create_dynamic_heatmap(config, genome):
    read_pairs = [
        LocationPair(
            Location([1, 1, 2, 3, 3], [8, 8, 1, 1, 1]),
            Location([1, 1, 1, 1, 3], [2, 8, 1, 3, 9])
        )
    ]
    creator = PreComputedDynamicHeatmapCreator(genome, config)
    heatmap = creator.get_dynamic_heatmap(read_pairs, gap_distance=0)
    print(heatmap)



location_a = Location([0, 1, 2], [3, 2, 1])
location_b = Location([1, 2, 0], [0, 1, 2])
location_pair = LocationPair(location_a, location_b)


def test_create_from_locations():
    dh = DynamicHeatmaps(np.array([4, 5, 6]), n_bins=3, scale_func=lambda x: x)
    dh.register_location_pairs(location_pair)
    start_to_start = dh.array[0, 0]
    assert np.sum(start_to_start) == 2
    assert start_to_start[1, 2, 2, 1] == 1
    assert start_to_start[2, 0, 1, 2] == 1
    end_to_start = dh.array[1, 0]
    assert np.sum(end_to_start) == 2
    assert end_to_start[0, 1, 0, 0] == 1
    assert end_to_start[1, 2, 2, 1] == 1
    assert dh.get_heatmap(Edge(NodeSide(0, 'r'),
                               NodeSide(1, 'l'))).array[0, 0] == 1


def test_heatmap_comparison():
    heatmaps = [DynamicHeatmap(a) for a in (
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]])]
    hc = HeatmapComparison(heatmaps)
    assert hc.locate_heatmap(DynamicHeatmap([[0, 0], [0, 0]])) == 0
    assert hc.locate_heatmap(DynamicHeatmap([[2, 4], [5, 7]])) == 1
    assert hc.locate_heatmap(DynamicHeatmap([[6, 7], [8, 9]])) == 2



