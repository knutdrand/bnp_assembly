import numpy as np
from numpy.testing import assert_array_equal

from bnp_assembly.location import LocationPair, Location
from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmap, PreComputedDynamicHeatmapCreator

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


