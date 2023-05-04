from bionumpy.genomic_data import GenomicLocation, Genome
from bnp_assembly.datatypes import GenomicLocationPair
from bnp_assembly.interaction_matrix import InteractionMatrix
import numpy as np
import pytest


@pytest.fixture
def genome():
    return Genome.from_dict({'A': 10,
                             'B': 20,
                             'C': 30})


@pytest.fixture
def genomic_locations_a(genome):
    return GenomicLocation.from_fields(genome.get_genome_context(),
                                       ['A', 'B'],
                                       [3, 4])


@pytest.fixture
def genomic_locations_b(genome):
    return GenomicLocation.from_fields(genome.get_genome_context(),
                                       ['B', 'C'],
                                       [5, 6])


@pytest.fixture
def location_pair(genomic_locations_a, genomic_locations_b):
    return GenomicLocationPair(genomic_locations_a, genomic_locations_b)


def test_interaction_matrix(location_pair):
    matrix = InteractionMatrix.from_locations_pair(location_pair)
    true_matrix = np.zeros((60, 60))
    coordinate_pairs = [(3, 15), (14, 36)]
    for pair in coordinate_pairs:
        true_matrix[pair] = 1
        true_matrix[pair[::-1]] = 1
    np.testing.assert_array_equal(matrix.data, true_matrix)


def test_interaction_matrix_binned(location_pair):
    bin_size = 2
    matrix = InteractionMatrix.from_locations_pair(location_pair, bin_size=bin_size)
    true_matrix = np.zeros((60//2, 60//2))
    coordinate_pairs = [(3//bin_size, 15//bin_size), (14//bin_size, 36//bin_size)]
    for pair in coordinate_pairs:
        true_matrix[pair] += 1
        true_matrix[pair[::-1]] += 1
    np.testing.assert_array_equal(matrix.data, true_matrix)

