import numpy as np
from numpy.testing import assert_array_equal
import pytest
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmap, PreComputedDynamicHeatmapCreator, \
    DynamicHeatmapConfig
import bionumpy as bnp

a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([2,4,6,8,10,12,14,16,18,20])


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
