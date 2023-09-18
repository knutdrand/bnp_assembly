import pytest

from bnp_assembly.distance_distribution import distance_dist
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.make_scaffold import get_forbes_counts, get_missing_region_counts
from bnp_assembly.missing_data import find_regions_with_missing_data, get_binned_read_counts, \
    find_regions_with_missing_data_from_bincounts, find_start_and_end_split_site_for_contig, \
    find_missing_regions_at_start_and_end_of_contigs
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
    counts = get_forbes_counts(next(read_pairs), contig_dict, cumulative_distribution, bin_size)
    bins, bin_sizes = get_missing_region_counts(contig_dict, next(read_pairs), bin_size)
    regions, reads_per_bp = find_regions_with_missing_data_from_bincounts(bin_size, bin_sizes, bins)
    assert regions == {0: [(10, 20)], 1: [(10, 20)]}


@pytest.fixture
def contig_sizes2():
    return {"contig1": 10, "contig2": 200, "contig3": 10}


@pytest.fixture
def missing_regions():
    return {
        "contig1": [(0, 4)],
        "contig3": [(0, 3), (5, 6), (6, 10)]
    }


def test_find_start_and_end_split_site_for_contig():
    assert find_start_and_end_split_site_for_contig(10, [(0, 2)]) == (2, 10)
    assert find_start_and_end_split_site_for_contig(10, [(0, 2), (2, 4)]) == (4, 10)
    assert find_start_and_end_split_site_for_contig(10, [(9, 10)]) == (0, 9)
    assert find_start_and_end_split_site_for_contig(10, [(7, 9), (9, 10)]) == (0, 7)
    assert find_start_and_end_split_site_for_contig(10, [(0, 3), (7, 9), (9, 10)]) == (3, 7)
    assert find_start_and_end_split_site_for_contig(10, [(0, 1), (1, 2), (4, 5), (7, 9), (9, 10)]) == (2, 7)
    assert find_start_and_end_split_site_for_contig(10, [(0, 5), (5, 10)]) == (0, 0)


def test_find_missing_start_end_contigs(contig_sizes2, missing_regions):
    contig_sizes = contig_sizes2
    splits = find_missing_regions_at_start_and_end_of_contigs(contig_sizes, missing_regions)

    assert splits == {
        "contig1": (4, 10),
        "contig2": (0, 200),
        "contig3": (3, 5)
    }

