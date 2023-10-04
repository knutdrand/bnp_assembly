import itertools

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from bnp_assembly.cli import estimate_max_distance2
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmap, PreComputedDynamicHeatmapCreator, \
    DynamicHeatmapConfig, find_bins_with_even_number_of_reads, get_dynamic_heatmap_config_with_even_bins, \
    get_all_possible_edges, get_gap_distances, find_bins_with_even_number_of_reads3
import bionumpy as bnp

from bnp_assembly.graph_objects import NodeSide, Edge
from bnp_assembly.location import Location, LocationPair
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmap, DynamicHeatmaps, HeatmapComparison, HeatmapComparisonRowColumns

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])


def test_create_from_positions():
    dh = DynamicHeatmap.create_from_positions((a, b), n_bins=3, scale_func=lambda x: x)
    assert np.sum(dh.array) == 1
    assert_array_equal(dh.array, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
    dh = DynamicHeatmap.create_from_positions((a, b), n_bins=5, scale_func=lambda x: x)
    assert_array_equal(dh.array,
                       np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))


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
    return DynamicHeatmapConfig(scale_func=lambda x: x, max_distance=4, n_bins=4)


@pytest.fixture
def genome():
    return {
        1: 10,
        2: 2,
        3: 20
    }


@pytest.fixture
def read_pairs():
    read_pairs = [
        LocationPair(
            Location([1, 1, 2, 3, 3], [8, 8, 1, 1, 1]),
            Location([1, 1, 1, 1, 3], [2, 8, 1, 3, 9])
        )
    ]
    return read_pairs


@pytest.mark.xfail
def test_create_dynamic_heatmap(config, genome, read_pairs):
    creator = PreComputedDynamicHeatmapCreator(genome, config, n_precomputed_heatmaps=2)
    assert creator._get_suitable_contigs_for_estimation() == [1, 3]
    heatmap = creator.get_dynamic_heatmap(read_pairs, gap_distance=0)
    print(heatmap)

    correct = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0.5],
        [0, 0, 0, 0]
    ])
    assert_array_equal(heatmap.array, correct)


# multiple heatmaps for different gaps
def test_create_dynamic_heatmaps(config, genome, read_pairs):
    read_pairs_stream = (read_pairs for _ in itertools.count())
    creator = PreComputedDynamicHeatmapCreator(genome, config, n_precomputed_heatmaps=4)
    heatmaps = creator.create(read_pairs_stream)
    assert len(heatmaps) == 4


location_a = Location([0, 1, 2], [3, 2, 1])
location_b = Location([1, 2, 0], [0, 1, 2])
location_pair = LocationPair(location_a, location_b)


def test_create_from_locations():
    dh = DynamicHeatmaps(np.array([4, 5, 6]), DynamicHeatmapConfig(n_bins=3, scale_func=lambda x: x, max_distance=3))
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


def test_create_from_many_locations():
    dh = DynamicHeatmaps(np.array([4, 5, 6]), DynamicHeatmapConfig(n_bins=3, scale_func=lambda x: x, max_distance=3))
    for _ in range(10000):
        dh.register_location_pairs(location_pair)

    assert dh.get_heatmap(Edge(NodeSide(0, 'r'),
                               NodeSide(1, 'l'))).array[0, 0] == 10000


def test_create_dynamic_heatmaps_with_binned_scale_func():
    distance_dist = np.cumsum(np.random.randint(0, 5, 20000))
    config = get_dynamic_heatmap_config_with_even_bins(distance_dist, n_bins=50, max_distance=10000)

    dh = DynamicHeatmaps(np.array([5000, 5000, 5000]), config)
    for _ in range(100):
        dh.register_location_pairs(LocationPair(Location([0], [4900]), Location([1], [500])))
        dh.register_location_pairs(LocationPair(Location([1], [500]), Location([0], [4900])))

    heatmap = dh.get_heatmap(Edge(NodeSide(0, 'r'), NodeSide(1, 'l'))).array
    print(heatmap)
    assert np.sum(heatmap) == 200


def test_heatmap_comparison():
    heatmaps = [DynamicHeatmap(a) for a in (
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]])]
    hc = HeatmapComparison(heatmaps)
    assert hc.locate_heatmap(DynamicHeatmap([[0, 0], [0, 0]])) == 0
    assert hc.locate_heatmap(DynamicHeatmap([[2, 4], [5, 7]])) == 1
    assert hc.locate_heatmap(DynamicHeatmap([[6, 7], [8, 9]])) == 2


def test_heatmap_comparison_row_cols():

    heatmap_stack = [
        DynamicHeatmap(
            np.zeros((3, 3))
        ),
        DynamicHeatmap(
            np.ones((3, 3)) * 2
        ),
        DynamicHeatmap(
            np.ones((3, 3)) * 10
        )
    ]

    heatmap = DynamicHeatmap([
        [1, 1, 1],  # 3
        [1, 2, 1],  # 4
        [2, 2, 1]   # 5
    ])

    comparison = HeatmapComparisonRowColumns.from_heatmap_stack(heatmap_stack)
    score = comparison.locate_heatmap(heatmap)
    assert score == 1

    heatmap = DynamicHeatmap([
        [1, 1],
    ])
    assert comparison.locate_heatmap(heatmap) == 1


@pytest.fixture
def distance_dist():
    return np.array([0, 1, 2, 3, 8, 10, 12, 13, 14, 100])


def test_find_bins_with_even_number_of_reads(distance_dist):
    splits = find_bins_with_even_number_of_reads(distance_dist, n_bins=3, max_distance=7)
    assert np.all(splits == [0, 4, 5, 7])


def test_find_bins_with_even_number_of_reads3():
    distance_dist = np.arange(20)
    splits = find_bins_with_even_number_of_reads3(distance_dist, n_bins=2, max_distance=10)
    assert_array_equal(splits, [0, 5, 10])


@pytest.mark.xfail  # fails after new even bin implementation
def test_get_hetmap_config_with_even_bins(distance_dist):
    config = get_dynamic_heatmap_config_with_even_bins(distance_dist, n_bins=3, max_distance=7)
    assert config.scale_func(0) == 0
    assert config.scale_func(4) == 1
    assert config.scale_func(5) == 2
    assert config.scale_func(6) == 2


def test_get_all_possible_edges():
    edges = list(get_all_possible_edges(5))
    assert max([edge.to_node_side.node_id for edge in edges]) == 4


def test_gap_distances():
    gaps = get_gap_distances(10000, 4)
    print(gaps)


def test_heatmap_comparison():
    stack = [
        DynamicHeatmap(
            np.array([
                [0, 0],
                [0, 0]
            ])
        ),
        DynamicHeatmap(
            np.array([
                [2, 2],
                [2, 2]
            ])
        ),
        DynamicHeatmap(
            np.array([
                [5, 5],
                [5, 5]
            ])
        )
    ]
    sample = DynamicHeatmap([
        [2, 0],
        [10, 10]
    ])

    comparison = HeatmapComparison(stack)
    assert comparison.locate_heatmap(sample) == 2

    sample = DynamicHeatmap([
        [100, 100],
        [100, 100]
    ])


def test_max_distance2():
    contig_sizes = [2, 50, 3, 5, 3]
    correct = 50 // 4
    #assert estimate_max_distance2(contig_sizes) == correct

    contig_sizes = [2, 1, 3, 100, 60, 1, 1]
    assert estimate_max_distance2(contig_sizes) == 100 // 4

