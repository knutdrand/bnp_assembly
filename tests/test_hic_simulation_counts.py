import numpy as np
import pytest
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.hic_distance_matrix import calculate_distance_matrices, count_window_combinastions
from bnp_assembly.path_finding import best_path


@pytest.fixture
def contig_list():
    return {0: 100, 1: 50}


@pytest.fixture
def location_pairs():
    location_a = Location.from_entry_tuples([
        (0, 10),
        (1, 40),
        (0, 8)])
    location_b = Location.from_entry_tuples([
        (1, 45),
        (0, 5),
        (0, 20)])
    return LocationPair(location_a, location_b)


@pytest.fixture
def contig_list2():
    return {0: 10, 1: 10, 2: 10}


@pytest.fixture
def location_pairs2():
    location_a = Location.from_entry_tuples([
        (1, 0),
        (2, 1)])
    location_b = Location.from_entry_tuples([
        (2, 9),
        (0, 0)])

    return LocationPair(location_a, location_b)


def test_count_window_combinations(contig_list, location_pairs):
    overlap_count, inside_counts = count_window_combinastions(contig_list, location_pairs)
    print(overlap_count)
    assert overlap_count[frozenset(((0, 'l'), (1, 'r')))] == 2
    assert sum(overlap_count.values()) == 2
    assert inside_counts[(0, 'l')] == 1


def test_distance_best_path(contig_list, location_pairs):
    graph = calculate_distance_matrices(contig_list, location_pairs)
    path = best_path(graph)
    a = path.to_list()
    print(a)
    b = path.reverse().to_list()
    correct_path = [(1, 0), (0, 0)]
    assert (a == correct_path or b == correct_path)


def test_count_window_combinations(contig_list2, location_pairs2):
    overlap_count, inside_counts = count_window_combinastions(contig_list2, location_pairs2, window_size=5)
    print(overlap_count)
    assert overlap_count[frozenset(((1, 'l'), (2, 'r')))] == 1
    assert overlap_count[frozenset(((2, 'l'), (0, 'l')))] == 1
    assert sum(overlap_count.values()) == 2
    assert sum(inside_counts.values()) == 0

def test_distance_distance_matrtix(contig_list2, location_pairs2):
    graph = calculate_distance_matrices(contig_list2, location_pairs2, window_size=5)
    data = graph.data
    assert np.argmin(data[2]) == 5, data


def test_distance_best_path2(contig_list2, location_pairs2):
    graph = calculate_distance_matrices(contig_list2, location_pairs2, window_size=5)
    path = best_path(graph)
    a = path.to_list()
    b = path.reverse().to_list()
    correct_path = [(1, 1), (2, 1), (0, 0)]
    assert (a == correct_path or b == correct_path)
