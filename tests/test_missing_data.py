import pytest
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.missing_data import find_regions_with_missing_data, get_binned_read_counts
import numpy as np


@pytest.fixture
def contig_dict():
    return {0: 30, 1: 40}


@pytest.fixture
def read_pairs():
    return LocationPair(
        Location.from_entry_tuples([
            (0, 5), (0, 25)
        ]),
        Location.from_entry_tuples([
            (1, 5), (1, 7), (1, 35)
        ])
    )


@pytest.fixture
def contig_dict_uneven():
    return {0: 30, 1: 45}


@pytest.fixture
def read_pairs2():
    return LocationPair(
        Location.from_entry_tuples([
            (0, 5), (0, 25)
        ]),
        Location.from_entry_tuples([
            (1, 5), (1, 7), (1, 15), (1, 25), (1, 35)
        ])
    )


def test_get_binned_read_counts(contig_dict, read_pairs):
    counts, bin_sizes  = get_binned_read_counts(10, contig_dict, read_pairs)
    assert np.all(counts[0] == [1, 0, 1])
    assert np.all(counts[1] == [2, 0, 0, 1])

    assert np.all(bin_sizes[0] == [10, 10, 10])
    assert np.all(bin_sizes[1] == [10, 10, 10, 10])


def test_get_binned_read_counts2(contig_dict_uneven, read_pairs2):
    counts, bin_sizes  = get_binned_read_counts(10, contig_dict_uneven, read_pairs2)
    assert np.all(bin_sizes[1] == [10, 10, 10, 10, 5])


def test_find_regions_with_missing_data(contig_dict, read_pairs):
    bin_size = 10
    regions = find_regions_with_missing_data(contig_dict, read_pairs, bin_size)

    correct = {0: [(10, 20)],
               1: [(10, 20), (20, 30)]}

    assert regions == correct


def test_find_regions_with_missing_data2(contig_dict_uneven, read_pairs2):
    bin_size = 10
    regions = find_regions_with_missing_data(contig_dict_uneven, read_pairs2, bin_size)

    correct = {
        0: [(10, 20)],
        1: [(40, 45)]
    }

