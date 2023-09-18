import pytest
from bnp_assembly.missing_data import find_start_and_end_split_site_for_contig, find_missing_regions_at_start_and_end_of_contigs


@pytest.fixture
def contig_sizes():
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


def test_find_missing_start_end_contigs(contig_sizes, missing_regions):
    splits = find_missing_regions_at_start_and_end_of_contigs(contig_sizes, missing_regions)

    assert splits == {
        "contig1": (4, 10),
        "contig2": (0, 200),
        "contig3": (3, 5)
    }

