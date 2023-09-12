import pytest

from bnp_assembly.distance_distribution import distance_dist
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.make_scaffold import process_reads
from bnp_assembly.missing_data import find_regions_with_missing_data, get_binned_read_counts, \
    find_regions_with_missing_data_from_bincounts
import numpy as np
from bnp_assembly.io import PairedReadStream

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
    regions, avg_coverage = find_regions_with_missing_data(contig_dict, read_pairs, bin_size)

    correct = {0: [(10, 20)],
               1: [(10, 20), (20, 30)]}

    assert regions == correct


def test_find_regions_with_missing_data2(contig_dict_uneven, read_pairs2):
    bin_size = 10
    regions, average_coverage = find_regions_with_missing_data(contig_dict_uneven, read_pairs2, bin_size)

    correct = {
        0: [(10, 20)],
        1: [(40, 45)]
    }

    assert regions == correct


@pytest.fixture
def read_pairs3():
    read_pairs = PairedReadStream.from_location_pair(
        LocationPair(
            Location.from_entry_tuples([
                (0, 5), (0, 25), (0, 3)
            ]),
            Location.from_entry_tuples([
                (1, 5), (1, 25), (0, 2)
            ])
        )
    )
    return read_pairs


def test_integration_from_read_pairs(read_pairs3):
    read_pairs = read_pairs3
    bin_size = 10
    contig_dict = {0: 30, 1: 30}
    cumulative_distribution = distance_dist(next(read_pairs), contig_dict)
    bins, bin_sizes, counts = process_reads(next(read_pairs), contig_dict, cumulative_distribution, bin_size)
    regions, reads_per_bp = find_regions_with_missing_data_from_bincounts(bin_size, bin_sizes, bins)
    assert regions == {0: [(10, 20)], 1: [(10, 20)]}
