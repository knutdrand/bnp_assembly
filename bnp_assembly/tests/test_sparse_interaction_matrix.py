import pytest
from bionumpy.genomic_data.global_offset import GlobalOffset
from bnp_assembly.io import PairedReadStream
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.sparse_interaction_matrix import NaiveSparseInteractionMatrix, BinnedNumericGlobalOffset, \
    SparseInteractionMatrix
import numpy as np
from numpy.testing import assert_array_equal


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


def test_from_reads(read_pairs3):
    contig_sizes = {0: 30, 1: 40}
    matrix = NaiveSparseInteractionMatrix.from_reads(contig_sizes, read_pairs3, 10)
    print(matrix.nonsparse_matrix)

    #assert_array_equal(
    #    matrix,
    #)


def test_binned_numeric_global_offset():
    contig_sizes = {0: 8, 1: 10, 2: 14}
    g = BinnedNumericGlobalOffset.from_contig_sizes(contig_sizes, 5)
    assert np.all(g._contig_n_bins == [2, 2, 3])
    assert np.all(g._contig_bin_offset == [0, 2, 4])

    assert np.all(g.from_local_coordinates(
        np.array([0, 1, 2]), np.array([0, 0, 0])) == [0, 2, 4])

    assert np.all(g.from_local_coordinates(
        np.array([0, 1, 2]), np.array([1, 1, 1])) == [0, 2, 4])

    assert np.all(g.from_local_coordinates(
        np.array([0, 1, 2]), np.array([7, 9, 13])) == [1, 3, 6])

    assert np.all(g.from_local_coordinates(
        np.array([0, 1, 2]), np.array([4, 5, 10])) == [1, 3, 6])

@pytest.fixture
def read_pairs2():
    read_pairs = PairedReadStream.from_location_pair(
        LocationPair(
            Location.from_entry_tuples([
                (0, 5), (0, 7), (0, 1)
            ]),
            Location.from_entry_tuples([
                (1, 5), (2, 13), (0, 2)
            ])
        )
    )
    return read_pairs


def test_sparse_interaction_matrix(read_pairs2):
    contig_sizes = {0: 8, 1: 10, 2: 14}
    g = BinnedNumericGlobalOffset.from_contig_sizes(contig_sizes, 5)
    matrix = SparseInteractionMatrix.from_reads(g, read_pairs2)
    nonsparse = matrix.nonsparse_matrix
    assert nonsparse[0, 0] == 2
    assert nonsparse[-1, 1] == 1
    nonbinned = matrix.to_nonbinned()


def test_nonbinned():
    contig_sizes = {0: 8, 1: 10, 2: 14}
    g = BinnedNumericGlobalOffset.from_contig_sizes(contig_sizes, 5)
    matrix = SparseInteractionMatrix.empty(g)
    size = g.total_size()
    matrix.sparse_matrix[0:size, 0:size] = 3
    nonbinned = matrix.to_nonbinned()
    assert np.sum(matrix.nonsparse_matrix) == np.sum(nonbinned.nonsparse_matrix)
    assert np.sum(matrix.sparse_matrix) == np.sum(nonbinned.sparse_matrix)


def test_get_unbinned_coordinates_from_global_binned_coordinates():
    contig_sizes = {0: 8, 1: 10, 2: 14}
    g = BinnedNumericGlobalOffset.from_contig_sizes(contig_sizes, 5)

    assert_array_equal(
        g.get_unbinned_coordinates_from_global_binned_coordinates(np.array([0, 2, 4])),
        np.array([0, 8, 18])
    )
    assert_array_equal(
        g.get_unbinned_coordinates_from_global_binned_coordinates(np.array([1, 3, 5])),
        np.array([4, 13, 23])
    )


def test_global_binned_conversion():
    """When converting from coordinate to a bin, and then back and forth one should get the same bin"""
    g = BinnedNumericGlobalOffset.from_contig_sizes({0: 13}, 5)
    for i in range(13):
        bin = g.from_local_coordinates(0, i)
        local = g.to_local_coordinates(bin)
        bin2 = g.from_local_coordinates(0, local)
        assert bin2 == bin

    g = BinnedNumericGlobalOffset.from_contig_sizes({0: 45983}, 50)
    for i in range(45983-50):
        bin = g.from_local_coordinates(0, i)
        local = g.to_local_coordinates(bin)
        bin2 = g.from_local_coordinates(0, local)
        assert bin2 == bin


def test_get_contig_submatrix():
    g = BinnedNumericGlobalOffset.from_contig_sizes({0: 13}, 5)
    matrix = SparseInteractionMatrix.empty(g)

    sub = matrix.get_contig_submatrix(0, 0, 1, 0, 1)
    assert sub.nonsparse_matrix.shape == (1, 1)

    sub = matrix.get_contig_submatrix(0, 0, 1, 5, 6)
    assert sub.nonsparse_matrix.shape == (1, 1)

    for i in range(6):
        for j in range(6):
            print(i, j)
            sub = matrix.get_contig_submatrix(0, i, i+1, j, j+1)
            assert sub.nonsparse_matrix.shape == (1, 1)

    for i in range(5):
        for j in range(5):
            sub = matrix.get_contig_submatrix(0, i, i+2, j, j+2)
            assert sub.nonsparse_matrix.shape == (1, 1)


def test_get_contig_submatrix2():
    g = BinnedNumericGlobalOffset.from_contig_sizes({0: 45983}, 50)
    matrix = SparseInteractionMatrix.empty(g)
    x_start = 24388
    x_end = 30135
    y_start = 15847
    y_end = 21594
    assert x_end-x_start == y_end-y_start
    sub = matrix.get_contig_submatrix(0, x_start, x_end, y_start, y_end)


def test_get_contig_submatrix3():
    g = BinnedNumericGlobalOffset.from_contig_sizes({0: 45979, 1: 45979, 2: 45979, 3: 45979}, 50)
    assert g.global_contig_offset(0) == 0
    assert g.global_contig_offset(1) == 45979
    assert g.global_contig_offset(3) == 45979*3

    print(g.contig_sizes)
    matrix = SparseInteractionMatrix.empty(g)
    x_start = 23187
    x_end = 28934
    y_start = 17044
    y_end = 22791
    assert x_end-x_start == y_end-y_start
    sub = matrix.get_contig_submatrix(3, x_start, x_end, y_start, y_end)


def test_get_contig_intra_matrix():
    g = BinnedNumericGlobalOffset.from_contig_sizes({0: 5, 1: 11, 2: 6}, 5)
    matrix = SparseInteractionMatrix.empty(g)
    matrix.add_one(0, 4, 0, 1)
    sub = matrix.get_contig_intra_matrix(0, 0, 5)
    assert_array_equal(sub.nonsparse_matrix, [[2]])
    print(sub)

    matrix.add_one(1, 2, 1, 1)
    sub = matrix.get_contig_intra_matrix(1, 0, 5)
    assert_array_equal(sub.nonsparse_matrix, [[2]])
    sub = matrix.get_contig_intra_matrix(1, 0, 11)
    assert_array_equal(sub.nonsparse_matrix, [
        [2, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])

