import numpy as np
from numpy.testing import assert_array_equal

from bnp_assembly.pre_sampled_dynamic_heatmap_comparison import DynamicHeatmap
a = np.array([1,2,3,4,5,6,7,8,9,10])
b = np.array([2,4,6,8,10,12,14,16,18,20])


def test_create_from_positions():
    dh = DynamicHeatmap.create_from_positions((a, b), n_bins=3, scale_func=lambda x: x)
    assert np.sum(dh.array) == 1
    assert_array_equal(dh.array, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]))
    dh = DynamicHeatmap.create_from_positions((a, b), n_bins=5, scale_func=lambda x: x)
    assert_array_equal(dh.array, np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0 ,0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0 ,0]]))
