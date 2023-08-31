import numpy as np
import plotly.express as px
from bnp_assembly.scaffold_splitting import count_possible_edge_pairs, count_edge_overlaps

import pytest


@pytest.fixture
def locations():
    return [2, 3, 5, 7, 11]


@pytest.fixture
def locations2():
    return [5, 6, 7, 8, 10]


@pytest.fixture
def pairs():
    return ([14, 1, 9, 4, 15, 15],
            [15, 2, 5, 4, 17, 18])


@pytest.fixture
def edge_locations():
    return [2, 4, 8, 16]


max_length = 4


def test_count_pairs(locations, edge_locations):
    edge_counts = count_possible_edge_pairs(locations, edge_locations, max_length)
    true_counts = [0, 1 + 2, 1, 0]
    assert edge_counts == true_counts


def test_count_pairs(locations2, edge_locations):
    edge_counts = count_possible_edge_pairs(locations2, edge_locations, max_length)
    true_counts = [0, 0, 1 + 2 + 2, 0]
    assert edge_counts == true_counts


def test_count_edge_pairs(pairs, edge_locations):
    locations_a, locations_b = pairs
    edge_counts = count_edge_overlaps(locations_a, locations_b, edge_locations, max_length)
    true_counts = [1, 0, 1, 2]
    np.testing.assert_array_equal(edge_counts, true_counts)


def test_random_locations():
    max_length = 30
    edge_locations = [10, 15, 20, 25, 30, 40]
    positions = [2, 3, 7, 7, 11, 13, 17, 19, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    locations_a = np.random.choice(positions, 1000)
    locations_b = np.random.choice(positions, 1000)
    edge_counts = count_edge_overlaps(locations_a, locations_b, edge_locations, max_length)
    possible = count_possible_edge_pairs(positions, edge_locations, max_length)
    for i in range(len(edge_locations)):
        assert possible[i] == sum(1 for position_a in positions for position_b in positions if position_a<edge_locations[i]<=position_b<=position_a+max_length)
    first_location = np.minimum(locations_a, locations_b)
    last_location = np.maximum(locations_a, locations_b)
    for i in range(len(edge_locations)):
        assert edge_counts[i] == np.sum((first_location < edge_locations[i]) & (last_location >= edge_locations[i]) & (last_location-first_location<=max_length))
    # assert edge_counts[0] == np.sum((first_location < 10) & (last_location >= 10) & (last_location-first_location<=10))



    print(edge_counts, possible)

    expected = np.array(possible)/(len(positions)**2/2) * len(locations_a)
    px.scatter(x =expected, y=edge_counts)


    # assert False
