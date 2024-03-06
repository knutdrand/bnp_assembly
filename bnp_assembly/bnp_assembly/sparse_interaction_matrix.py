import logging
from typing import Dict, Union, Tuple, List

import pandas as pd
import plotly.express as px
import matspy
import scipy.sparse
from bionumpy.genomic_data.global_offset import GlobalOffset
import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter
from tqdm import tqdm

from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.distance_distribution import DistanceDistribution
from bnp_assembly.graph_objects import Edge
from bnp_assembly.io import PairedReadStream
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import sklearn
import matplotlib.pyplot as plt


class NumericGlobalOffset(GlobalOffset):
    def __init__(self, contig_sizes):
        self._sizes = np.array(list(contig_sizes.values()), dtype=int)
        self._offset = np.insert(np.cumsum(self._sizes), 0, 0)

    def get_offset(self, contig_ids):
        return self._offset[contig_ids]

    def get_size(self, contig_ids):
        return self._sizes[contig_ids]

    def from_local_coordinates(self, contig_ids, local_offsets):
        mask = local_offsets >= self.get_size(contig_ids)
        if np.any(np.atleast_1d(mask)):
            raise Exception('Coordinate outside of reference:', local_offsets[mask], self.get_size(contig_ids)[mask])
        coordinates = self.get_offset(contig_ids) + local_offsets
        return coordinates


class BinnedNumericGlobalOffset:
    """
    Can be used to find a global bin offset with varying bin size per contig
    All coordinates returned are bins, not positions
    """
    def __init__(self, contig_sizes: np.ndarray, contig_n_bins: np.ndarray, contig_bin_offset: np.ndarray):
        self._contig_n_bins = contig_n_bins
        self._contig_bin_offset = contig_bin_offset
        self._contig_sizes = contig_sizes

    def total_size(self):
        return self._contig_n_bins.sum()

    @property
    def contig_sizes(self):
        return self._contig_sizes

    @classmethod
    def from_contig_sizes(cls, contig_sizes: Dict[int, int], approx_bin_size: int):
        assert np.all(list(contig_sizes.keys()) == np.arange(len(contig_sizes)))

        # find number of bins per contig that makes each bin close to approx_bin_size
        contig_n_bins = np.zeros(len(contig_sizes), dtype=int)
        contig_bin_offset = np.zeros(len(contig_sizes), dtype=int)
        offset = 0
        for contig_id, size in contig_sizes.items():
            extended_size = (size + approx_bin_size-1)
            extended_size -= extended_size % approx_bin_size
            n_bins = extended_size // approx_bin_size
            contig_n_bins[contig_id] = n_bins
            contig_bin_offset[contig_id] = offset
            offset += n_bins

        contig_sizes = np.array(list(contig_sizes.values()), dtype=int)
        return cls(contig_sizes, contig_n_bins, contig_bin_offset)

    def get_offset(self, contig_ids):
        return self._contig_bin_offset[contig_ids]

    def from_local_coordinates(self, contig_ids: np.ndarray, local_offsets: np.ndarray):
        local_bin_offsets = (local_offsets / self._contig_sizes[contig_ids]) * self._contig_n_bins[contig_ids]
        coordinates = self.get_offset(contig_ids) + local_bin_offsets.astype(int)
        assert np.all(coordinates < self.total_size()), (f"Coordinates {coordinates} exceed total size {self.total_size()}. "
                                                       f"Contig ids: {contig_ids}, \n"
                                                       f"local_offsets: {local_offsets}, \n"
                                                       f"contig_sizes: {self._contig_sizes}, \n"
                                                       f"contig_n_bins: {self._contig_n_bins}, \n"
                                                       f"contig_bin_offset: {self._contig_bin_offset}")
        return coordinates

    def to_local_coordinates(self, global_coordinates):
        return self.get_unbinned_coordinates_from_global_binned_coordinates(global_coordinates)

    def global_contig_offset(self, contig_id):
        cumulative_contig_sizes = np.insert(np.cumsum(self._contig_sizes), 0, 0)
        return cumulative_contig_sizes[contig_id]

    def get_unbinned_coordinates_from_global_binned_coordinates(self, global_binned_coordinates: np.ndarray, as_float=False):
        assert np.all(global_binned_coordinates < self.total_size()), (global_binned_coordinates[global_binned_coordinates >= self.total_size()], self.total_size())
        # returns a tuple of contig_ids and local_offsets
        contig_ids = np.searchsorted(self._contig_bin_offset, global_binned_coordinates, side='right') - 1
        local_offsets = (global_binned_coordinates - self._contig_bin_offset[contig_ids]) * self._contig_sizes[contig_ids] / self._contig_n_bins[contig_ids]
        assert np.all(local_offsets < self._contig_sizes[contig_ids]), (local_offsets, self._contig_sizes[contig_ids])
        #cumulative_contig_sizes = np.insert(np.cumsum(self._contig_sizes), 0, 0)
        coordinates = self.global_contig_offset(contig_ids) + local_offsets
        if as_float:
            return coordinates
        return np.ceil(coordinates)
        #return (cumulative_contig_sizes[contig_ids] + local_offsets).astype(int)

    def get_unbinned_local_coordinates_from_contig_binned_coordinates(self, contig_id, binned_coordinates: np.ndarray, as_float=False):
        """Translates a binned coordinate to a real offset at the contig"""
        assert np.all(binned_coordinates >= 0)
        coordinates = binned_coordinates * (self._contig_sizes[contig_id] / self._contig_n_bins[contig_id])  # last division first to prevent overflow
        assert np.all(coordinates >= 0), (coordinates, binned_coordinates, self._contig_sizes[contig_id], self._contig_n_bins[contig_id])
        assert np.all(coordinates < self._contig_sizes[contig_id])
        if as_float:
            return coordinates
        return np.ceil(coordinates)

    def round_global_coordinate(self, contig_id, local_offset, as_float=False):
        """
        "Rounds" coordinate down to nearest bin and then translates back
        """
        bin = self.from_local_coordinates(contig_id, local_offset)
        global_offset = self.get_unbinned_coordinates_from_global_binned_coordinates(bin, as_float=as_float)
        new = global_offset-self.global_contig_offset(contig_id)
        # should not change bin
        if not as_float:
            assert self.from_local_coordinates(contig_id, new) == bin
        return new

    def distance_to_n_bins(self, contig_id, distance):
        """
        Returns number of bins that the distance corresponds to
        """
        contig_binsize = self._contig_sizes[contig_id] / self._contig_n_bins[contig_id]
        return distance // contig_binsize

    def contig_first_bin(self, contig_id):
        return self.from_local_coordinates(contig_id, 0)

    def contig_last_bin(self, contig_id, inclusive=True):
        if inclusive:
            return self.from_local_coordinates(contig_id, self.contig_sizes[contig_id]-1)
        return self.from_local_coordinates(contig_id, self.contig_sizes[contig_id]-1) + 1


class NaiveSparseInteractionMatrix:
    """
    Using scipy sparse matrix to represent a heatmap with binned counts
    """
    def __init__(self, data: np.ndarray, global_offset: NumericGlobalOffset, bin_size: int):
        self._data = data
        self._global_offset = global_offset
        self._bin_size = bin_size

    @property
    def sparse_matrix(self):
        return self._data

    @property
    def nonsparse_matrix(self):
        return self._data.toarray()

    @classmethod
    def from_reads(cls, contig_sizes, reads: PairedReadStream, bin_size=100):
        reads = next(reads)
        global_offset = NumericGlobalOffset(contig_sizes)
        size = (global_offset.total_size() + bin_size - 1) // bin_size
        matrix = lil_matrix((size, size), dtype=float)

        n_added = 0
        for chunk in reads:
            logging.info(f'{n_added} reads added')
            n_added += len(chunk.location_a)
            global_pair = tuple(global_offset.from_local_coordinates(l.contig_id, l.offset) // bin_size
                                for l in (chunk.location_a, chunk.location_b))
            #np.add.at(matrix, global_pair, 1)
            #np.add.at(matrix, global_pair[::-1], 1)
            for a, b in zip(*global_pair):
                matrix[a, b] += 1
                matrix[b, a] += 1

        logging.info("Convert to row sparse")
        matrix = matrix.tocsr()
        logging.info("Converted to row sparse")

        return cls(matrix, global_offset, bin_size)


class SparseInteractionMatrix(NaiveSparseInteractionMatrix):
    """
    Only relies on a global offset class for translation of coordinates,
    Can then used BinnedGlobalOffset
    """
    def __init__(self, data: np.ndarray, global_offset: Union[NumericGlobalOffset, BinnedNumericGlobalOffset],
                 allow_nonsymmetric=False):
        if not allow_nonsymmetric:
            assert (data.T != data).nnz == 0, "Matrix must be symmetric"
        self._data = data
        self._global_offset = global_offset

    @property
    def n_contigs(self):
        return len(self._global_offset.contig_sizes)

    @property
    def contig_n_bins(self):
        return self._global_offset._contig_n_bins

    @classmethod
    def empty(cls, global_offset: Union[NumericGlobalOffset, BinnedNumericGlobalOffset], allow_nonsymmetric=False):
        size = global_offset.total_size()
        matrix = lil_matrix((size, size), dtype=float)
        return cls(matrix, global_offset, allow_nonsymmetric=allow_nonsymmetric)

    def set_matrix(self, matrix):
        assert self._data.shape == matrix.shape
        self._data = matrix

    def add_one(self, contig_a, offset_a, contig_b, offset_b):
        x = self._global_offset.from_local_coordinates(contig_a, offset_a)
        y = self._global_offset.from_local_coordinates(contig_b, offset_b)
        self._data[x, y] += 1
        self._data[y, x] += 1

    @classmethod
    def from_reads(cls, global_offset: BinnedNumericGlobalOffset, reads: PairedReadStream):
        reads = next(reads)
        size = global_offset.total_size()
        logging.info(f"Making empty sparse matrix of size {size}")
        matrix = lil_matrix((size, size), dtype=float)

        n_added = 0
        for chunk in reads:
            logging.info(f'{n_added} reads added')
            n_added += len(chunk.location_a)
            global_pair = tuple(global_offset.from_local_coordinates(l.contig_id, l.offset)
                                for l in (chunk.location_a, chunk.location_b))
            for a, b in zip(*global_pair):
                matrix[a, b] += 1
                matrix[b, a] += 1

        logging.info("Convert to row sparse")
        matrix = matrix.tocsr()
        logging.info("Convert to row sparse")

        return cls(matrix, global_offset)

    def get_submatrix(self, x_slice, y_slice):
        return self._data[x_slice, y_slice]

    def flip_rows(self):
        self._data = self._data[::-1, :]

    def get_contig_intra_matrix(self, contig_id: int, start: int, end: int) -> 'SparseInteractionMatrix':
        # returns a sparse matrix of intra-matrix at contig from start to end position relative to contig start
        start_bin = self._global_offset.from_local_coordinates(contig_id, start)
        end_bin = self._global_offset.from_local_coordinates(contig_id, end)
        submatrix = self.get_submatrix(slice(start_bin, end_bin), slice(start_bin, end_bin))

        new_global_offset = BinnedNumericGlobalOffset(np.array([end-start]),
                                                      np.array([end_bin-start_bin]),
                                                      np.array([0]))
        return SparseInteractionMatrix(submatrix, new_global_offset)

    def get_edge_interaction_matrix(self, edge: Edge, orient_according_to_nearest_interaction=True) -> np.ndarray:
        """
        returns submatrix of interaction for a given Edge.
        First row column in returned matrix represent nearest interaction
        """
        node_a = edge.from_node_side.node_id
        node_b = edge.to_node_side.node_id

        # get whole part of interaction matrix, flip it correctly according to node sides
        # so that nearest interactions are in first row/column
        g = self._global_offset
        matrix = self._data[
            g.contig_first_bin(node_a):g.contig_last_bin(node_a)+1,
            g.contig_first_bin(node_b):g.contig_last_bin(node_b)+1
        ]

        if orient_according_to_nearest_interaction:
            if edge.from_node_side.side == 'l':
                if edge.to_node_side.side == 'l':
                    return matrix
                else:
                    return matrix[:, ::-1]
            else:
                if edge.to_node_side.side == 'l':
                    return matrix[::-1, :]
                else:
                    return matrix[::-1, ::-1]
        else:
            if edge.from_node_side.side == 'l':
                matrix = matrix[::-1, :]
            if edge.to_node_side.side == 'r':
                matrix = matrix[:, ::-1]
            return matrix

    def get_unbinned_edge_interaction_matrix_as_indexes_and_values(self, edge: Edge) -> np.ndarray:
        """
        Returns with unbinned coordinates
        """
        matrix = self.get_edge_interaction_matrix(edge)
        # translate nonnegative coordinates. The two nodes can have different binning
        new_size = (self._global_offset.contig_sizes[edge.from_node_side.node_id],
                    self._global_offset.contig_sizes[edge.to_node_side.node_id])
        #unbinned = np.zeros(new_size)
        rows, cols = matrix.nonzero()
        values = np.array(matrix[rows, cols]).ravel()
        new_rows = self._global_offset.get_unbinned_local_coordinates_from_contig_binned_coordinates(edge.from_node_side.node_id, rows)
        new_cols = self._global_offset.get_unbinned_local_coordinates_from_contig_binned_coordinates(edge.to_node_side.node_id, cols)
        #unbinned[new_rows, new_cols] = values
        return new_rows, new_cols, values
        #return unbinned


    def get_contig_submatrix(self, contig_id: int, x_start: int, x_end: int, y_start: int, y_end: int) -> 'SparseInteractionMatrix':
        assert x_end-x_start == y_end-y_start
        assert y_end <= self._global_offset.contig_sizes[contig_id], (y_end, self._global_offset.contig_sizes[contig_id])
        assert x_end <= self._global_offset.contig_sizes[contig_id], (x_end, self._global_offset.contig_sizes[contig_id])
        assert x_start >= 0
        assert y_start >= 0

        """
        "Round" coordinates down to nearest bin before converting to bins, to ensure that the submatrix
        is quadratic. If not, one direction may have more bins than the other when x_end-x_start is the same as y_end-y_start
        """
        #logging.info(f"before x_start: {x_start}, x_end: {x_end}, y_start: {y_start}, y_end: {y_end}")
        contig_size = self._global_offset.contig_sizes[contig_id]
        x_size = x_end-x_start
        y_size = y_end-y_start
        x_start = self._global_offset.round_global_coordinate(contig_id, x_start, as_float=True)
        y_start = self._global_offset.round_global_coordinate(contig_id, y_start, as_float=True)

        x_bin_start = self._global_offset.from_local_coordinates(contig_id, x_start)
        y_bin_start = self._global_offset.from_local_coordinates(contig_id, y_start)+1  # +1 because start is exlusive (other direction)

        # end bins will be exclusive, i.e. first bin where first position is at end
        x_bin_end = int(x_bin_start + self._global_offset.distance_to_n_bins(contig_id, x_size))
        y_bin_end = int(y_bin_start + self._global_offset.distance_to_n_bins(contig_id, y_size))

        assert y_bin_end-y_bin_start == x_bin_end-x_bin_start
        submatrix = self.get_submatrix(slice(y_bin_start, y_bin_end), slice(x_bin_start, x_bin_end))

        #logging.info(f"New contig size: {x_end-x_start}, new bin size: {bins[1]-bins[0]}. Y: {y_start}, {y_end}, {y_end-y_start}. Bins: {bins}")
        new_global_offset = BinnedNumericGlobalOffset(np.array([x_size]),
                                                      np.array([x_bin_end-x_bin_start]),
                                                      np.array([0]))
        return SparseInteractionMatrix(submatrix, new_global_offset, allow_nonsymmetric=True)

    def to_nonbinned(self, return_indexes_and_values=False):
        """
        Returns a new SparseInteractionMatrix with a NumericGlobalOffset, i.e. no binning.
        Much faster with return_indexes_and_values, which does not create a matrix
        """
        contig_sizes = {i: size for i, size in enumerate(self._global_offset.contig_sizes)}

        # get all indexes and values from old matrix, translate indexes to new coordinates
        y, x = self._data.nonzero()
        assert np.all(y < self._global_offset.total_size())
        new_x = self._global_offset.get_unbinned_coordinates_from_global_binned_coordinates(x)
        new_y = self._global_offset.get_unbinned_coordinates_from_global_binned_coordinates(y)

        if return_indexes_and_values:
            return new_y.astype(int), new_x.astype(int), np.array(self._data[y, x]).ravel()

        new_global_offset = NumericGlobalOffset(contig_sizes)
        new_matrix = SparseInteractionMatrix.empty(new_global_offset)
        assert np.all(new_x < new_matrix._data.shape[1]), (new_x[new_x >= new_matrix._data.shape[1]], new_matrix._data.shape[1])
        assert np.all(new_y < new_matrix._data.shape[0]), (new_y[new_y >= new_matrix._data.shape[0]], new_matrix._data.shape[0])
        new_matrix._data[new_y, new_x] = self._data[y, x]
        return new_matrix

    def contig_bin_size(self, contig):
        return self._global_offset.contig_sizes[contig] / self._global_offset._contig_n_bins[contig]

    def get_contig_coverage_counts(self, contig):
        submatrix = self.contig_submatrix(contig)
        return np.array(np.sum(submatrix, axis=0))[0]

    def contig_submatrix(self, contig):
        start = self._global_offset.contig_first_bin(contig)
        end = self._global_offset.contig_last_bin(contig)
        submatrix = self._data[start:end, start:end]
        return submatrix

    def mean_coverage_per_bin(self):
        n_reads = self._data.sum()
        n_bins = self._data.shape[0]
        return n_reads / n_bins

    def trim_with_clips(self, contig_clips: Dict[int, Tuple[int, int]]):
        """Removes bins at ends of conigs that are clipped. Assumes clipping matches bins"""
        new_contig_sizes = self._global_offset.contig_sizes.copy()
        new_contig_n_bins = self._global_offset._contig_n_bins.copy()
        bin_sizes = {self.contig_bin_size(contig) for contig in contig_clips}

        bins_to_remove = np.zeros(self._data.shape[0], bool)

        for contig, (start, end) in contig_clips.items():
            contig_start_bin = self._global_offset.contig_first_bin(contig)
            contig_end_bin = self._global_offset.contig_last_bin(contig)
            # start is where clipping at start end, end is where clipping at end begins
            start_bin = self._global_offset.from_local_coordinates(contig, start)
            assert start_bin >= contig_start_bin
            if start_bin > contig_start_bin:
                bins_to_remove[contig_start_bin:start_bin] = True
                new_contig_n_bins[contig] -= start_bin-contig_start_bin
                new_contig_sizes[contig] -= start

            if end < self._global_offset.contig_sizes[contig]:
                end_bin = self._global_offset.from_local_coordinates(contig, end)
                assert end_bin <= contig_end_bin
                # contig_end_bin is last bin in contig, not exclusive ?
                bins_to_remove[end_bin:contig_end_bin+1] = True
                new_contig_n_bins[contig] -= contig_end_bin+1-end_bin
                new_contig_sizes[contig] -= self._global_offset.contig_sizes[contig]-end

        logging.info(f"Old contig sizes: {self._global_offset.contig_sizes}, old contig n bins: {self._global_offset._contig_n_bins}")
        logging.info(f"New contig sizes: {new_contig_sizes}, new contig n bins: {new_contig_n_bins}")

        logging.info(f"Removing {np.sum(bins_to_remove)}/{len(bins_to_remove)} bins")
        self._data = self._data[~bins_to_remove, :][:, ~bins_to_remove]
        assert self._data.shape[0] == np.sum(~bins_to_remove), f"{self._data.shape}, {np.sum(~bins_to_remove)}"
        assert self._data.shape[0] == self._data.shape[1]
        new_contig_offsets = np.insert(np.cumsum(new_contig_n_bins), 0, 0)
        self._global_offset = BinnedNumericGlobalOffset(new_contig_sizes, new_contig_n_bins, new_contig_offsets)

    def normalize(self):
        """Note: This does not keep symmetry"""
        logging.info("Normalizing %s" % type(self._data))
        old_row_mean = np.sum(self._data) / self._data.shape[0]
        self._data = sklearn.preprocessing.normalize(self._data, axis=0, norm='l1')
        self._data = sklearn.preprocessing.normalize(self._data, axis=1, norm='l1')
        self._data *= old_row_mean

    def normalize_on_row_and_column_products(self):
        row_sums = np.array(self._data.sum(axis=1)).ravel()
        column_sums = np.array(self._data.sum(axis=0)).ravel()
        rows, cols = np.nonzero(self.sparse_matrix)
        weights = row_sums[rows] + column_sums[cols]
        #weights = np.maximum(row_sums[rows], column_sums[cols])
        mean_weight = np.mean(weights)
        weights = weights.astype(float)
        weights /= float(mean_weight)
        self._data = self._data.astype(float)
        self._data.data /= weights
        self.assert_is_symmetric()

    def average_read_pair_distance(self):
        read1, read2 = np.nonzero(self._data)
        pos1 = self._global_offset.get_unbinned_coordinates_from_global_binned_coordinates(read1, as_float=True)
        pos2 = self._global_offset.get_unbinned_coordinates_from_global_binned_coordinates(read2, as_float=True)
        total = np.sum(self._data)
        distances = np.abs(pos1-pos2)
        weights = np.array(self._data[read1, read2]).ravel()
        return np.sum(distances*weights) / total

    def assert_is_symmetric(self):
        assert (self.sparse_matrix.T != self.sparse_matrix).nnz == 0

    def get_matrix_for_path(self, contigs: List[DirectedNode], as_raw_matrix=True):
        if np.all([contig.node_id for contig in contigs] == np.arange(len(contigs))) and all([contig.orientation == '+' for contig in contigs]):
            # path is same as matrix, return as is
            return self._data if as_raw_matrix else self
        rows = []
        for contig1 in contigs:
            columns_in_row = []
            for contig2 in contigs:
                edge = Edge(contig1.right_side, contig2.left_side)
                submatrix = self.get_edge_interaction_matrix(edge, orient_according_to_nearest_interaction=False)
                columns_in_row.append(submatrix)

            rows.append(scipy.sparse.hstack(columns_in_row))
        matrix = scipy.sparse.vstack(rows)

        # should be symmetric if everything went correctly
        #mismatch = (matrix.T != matrix).nnz
        #assert mismatch == 0, mismatch

        if as_raw_matrix:
            return matrix
        else:
            # return as SparseInteractionMatrix, need to make a new global offset
            new_contigs = [contig.node_id for contig in contigs]
            new_contig_sizes = np.array([self._global_offset.contig_sizes[contig] for contig in new_contigs])
            new_contig_n_bins = np.array([self._global_offset._contig_n_bins[contig] for contig in new_contigs])
            new_contig_offsets = np.insert(np.cumsum(new_contig_n_bins), 0, 0)
            new_global_offset = BinnedNumericGlobalOffset(new_contig_sizes, new_contig_n_bins, new_contig_offsets)
            return SparseInteractionMatrix(matrix, new_global_offset)

    def flip_contig(self, contig):
        logging.info("Flipping contig")
        start = self._global_offset.contig_first_bin(contig)
        end = self._global_offset.contig_last_bin(contig, inclusive=False)
        self._data[start:end, :] = self._data[start:end, :][::-1, :]
        self._data[:, start:end] = self._data[:, start:end][:, ::-1]
        logging.info("Flipped contig")

    def plot(self, xaxis_names=None):
        return self.plot_submatrix(0, self.n_contigs-1, xaxis_names=xaxis_names)

    def plot_submatrix(self, from_contig: int, to_contig: int, xaxis_names=None):
        start = self._global_offset.contig_first_bin(from_contig)
        end = self._global_offset.contig_last_bin(to_contig, inclusive=False)
        matrix = self._data[start:end, start:end]
        logging.info(f"Total matrix size: {matrix.shape}")
        if xaxis_names is None:
            xaxis_names = [str(c) for c in range(from_contig, to_contig+1)]
        xaxis_names = [x.replace("contig", "") for x in xaxis_names]

        offsets = [self._global_offset.contig_first_bin(c) - start for c in range(from_contig, to_contig+1)]
        buckets = max(200, min(2000, matrix.shape[0]//500))
        logging.info(f"Number of buckets: {buckets}")
        fig, ax = matspy.spy_to_mpl(matrix, buckets=buckets, figsize=10, shading='relative')
        plt.vlines(offsets, 0, matrix.shape[0], color='b')
        plt.hlines(offsets, 0, matrix.shape[0], color='b')
        ax.set_xticks(offsets)
        ax.set_xticklabels(xaxis_names)
        return fig, ax

    def to_lil_matrix(self):
        self._data = self._data.tolil()

    def to_csr_matrix(self):
        self._data = self._data.tocsr()

    def get_subset_on_contigs(self, first_contig, last_contig):
        start = self._global_offset.contig_first_bin(first_contig)
        end = self._global_offset.contig_last_bin(last_contig, inclusive=False)
        new_matrix = self._data[start:end, start:end]
        new_contig_sizes = self._global_offset.contig_sizes[first_contig:last_contig+1]
        new_contig_n_bins = self._global_offset._contig_n_bins[first_contig:last_contig+1]
        new_contig_offsets = np.insert(np.cumsum(new_contig_n_bins), 0, 0)
        new_global_offset = BinnedNumericGlobalOffset(new_contig_sizes, new_contig_n_bins, new_contig_offsets)
        return SparseInteractionMatrix(new_matrix, new_global_offset)

    def set_values_below_threshold_to_zero(self, threshold):
        self._data.data[self._data.data < threshold] = 0
        self._data.eliminate_zeros()

    @classmethod
    def from_np_matrix(cls, global_offset, matrix) -> 'SparseInteractionMatrix':
        return cls(csr_matrix(matrix), global_offset)


    def median_weight(self):
        return np.median(np.array(self._data.data).ravel())

    def get_reads_crossing_all_positions(self, max_distance_between_reads=100000):
        m = self.sparse_matrix
        size = m.shape[0]
        rows, cols = m.nonzero()
        mask = np.abs(rows-cols) <= max_distance_between_reads
        rows = rows[mask]
        cols = cols[mask]

        diagonal_index = rows + cols
        diagonal_values = np.zeros(m.shape[0] * 2 - 1)
        values = np.array(m[rows, cols]).ravel()
        diagonal_values = np.bincount(diagonal_index, weights=values, minlength=diagonal_values.shape[0])
        diagonal_size = np.concatenate([np.arange(1, size), [size], np.arange(size - 1, 0, -1)])
        diagonal_values = diagonal_values #/ diagonal_size
        return diagonal_values

    def left_inter_matrix(self, contig, max_distance):
        """Gives inter-matrix to the left of the contig to the max distance, and contig-distance to the right"""
        contig_start = self._global_offset.contig_first_bin(contig)
        contig_end = self._global_offset.contig_last_bin(contig, inclusive=False)

    def edge_score(self, index, minimum_assumed_chromosome_size: int=1000, background_matrix: 'BackgroundMatrix' = None):
        """Returns an edge score for splitting at the given index.
        The scores is minimum of the score of a right window and left window.
        For one window, we never look further than the previous contig sizes (in case we end up outside a chromosome)
        For the other window, we never look further than the next contig size

        If background_matrix is given, score is divided by score in same sized region in background matrix
        """
        assert index > 0
        assert index < self.n_contigs
        if background_matrix:
            # never look longer than background matrix
            minimum_assumed_chromosome_size = min(background_matrix.matrix.shape[1], minimum_assumed_chromosome_size)

        prev_contig_end = self._global_offset.contig_last_bin(index-1, inclusive=False)
        next_contig_start = self._global_offset.contig_first_bin(index)
        ystart = max(0, prev_contig_end-minimum_assumed_chromosome_size)
        yend = prev_contig_end
        xstart = next_contig_start
        xend = min(next_contig_start+minimum_assumed_chromosome_size, self.sparse_matrix.shape[1])
        matrix = self.sparse_matrix[ystart:yend, xstart:xend]

        prev_contig_size = self._global_offset.contig_sizes[index-1]
        next_contig_size = self._global_offset.contig_sizes[index]
        matrix1 = matrix[-prev_contig_size:, :]  # never go further back than the prev
        matrix2 = matrix[:, :next_contig_size]  # never go further than the next

        demominator1 = (matrix1.shape[1]*matrix1.shape[0])
        demominator2 = (matrix2.shape[1]*matrix2.shape[0])
        if background_matrix is not None:
            demominator1 = background_matrix.get_score(matrix1.shape[0], matrix1.shape[1])
            demominator2 = background_matrix.get_score(matrix2.shape[0], matrix2.shape[1])
            assert demominator1 > 0
            assert demominator2 > 0

        score1 = np.sum(matrix1) / demominator1
        score2 = np.sum(matrix2) / demominator2
        return max(score1, score2)




def average_element_distance(sparse_matrix):
    value = total_element_distance(sparse_matrix)
    total = np.sum(sparse_matrix)
    return value / total


def total_element_distance(sparse_matrix, max_distance_to_consider: int=None):
    pos1, pos2 = np.nonzero(sparse_matrix)
    distances = np.abs(pos1 - pos2)
    if max_distance_to_consider is not None:
        mask = distances <= max_distance_to_consider
        distances = distances[mask]
        pos1 = pos1[mask]
        pos2 = pos2[mask]

    weights = np.array(sparse_matrix[pos1, pos2]).ravel()
    # divide by two since matrix contains each pair twice (below and above diagonal)
    return np.sum(distances * weights) // 2



class LogProbSumOfReadDistances:
    def __init__(self, pmf: np.ndarray):
        assert np.all(pmf) > 0
        self._pmf = pmf

    def __call__(self, interaction_matrix):
        """
        Finds the sum of the log of probabilities of observing the given (binned) read distances
        distance_probabilities is a pmf array (with same binning)
        """
        assert len(self._pmf) > interaction_matrix.shape[0]
        pos1, pos2 = np.nonzero(interaction_matrix)
        distances = np.abs(pos1 - pos2)
        weights = np.array(interaction_matrix[pos1, pos2]).ravel()
        assert np.all(weights) > 0
        probs = self._pmf[distances]
        assert np.isnan(probs).sum() == 0
        assert np.all(probs) > 0
        result = np.sum(probs * weights)
        assert np.isnan(result).sum() == 0
        return result


def estimate_distance_pmf_from_sparse_matrix(sparse_matrix: SparseInteractionMatrix) -> DistanceDistribution:
    # use largestt contig to estimate
    contig_sizes = sparse_matrix._global_offset._contig_n_bins
    largest_contig = np.argmax(contig_sizes)
    logging.info(f"Using contig {largest_contig} to estimate distance pmf")
    submatrix = sparse_matrix.contig_submatrix(largest_contig)
    pos1, pos2 = np.nonzero(submatrix)
    distances = np.abs(pos1 - pos2)
    weights = np.array(submatrix[pos1, pos2]).ravel()
    pmf = np.bincount(distances, weights=weights)
    pmf = pmf / np.sum(pmf)
    dist = DistanceDistribution(pmf)
    dist.smooth()
    return dist

def estimate_distance_pmf_from_sparse_matrix2(sparse_matrix: SparseInteractionMatrix) -> DistanceDistribution:
    # find conigs that contribute to at least some percentage of the genome
    contig_sizes = sparse_matrix._global_offset._contig_n_bins
    contigs_to_use = contigs_covering_percent_of_total(contig_sizes)
    logging.info(f"Using contigs {contigs_to_use} to estimate distance pmf")

    max_distance = min([contig_sizes[contig] for contig in contigs_to_use])
    logging.info(f"Max distance: {max_distance}")

    pmf = np.zeros(max_distance)

    for contig in contigs_to_use:
        submatrix = sparse_matrix.contig_submatrix(contig)
        pos1, pos2 = np.nonzero(submatrix)
        distances = np.abs(pos1 - pos2)
        mask = distances < max_distance
        distances = distances[mask]
        weights = np.array(submatrix[pos1, pos2]).ravel()[mask]
        pmf += np.bincount(distances, weights=weights, minlength=max_distance)

    # make distance max of previous cumulative to avoid zeros and remove noise
    unfiltered = pmf.copy()
    pmf[-1] = np.min(pmf[pmf > 0])
    filtered = uniform_filter1d(pmf, size=20)
    pmf[20:] = filtered[20:]   # do not filter close signals, these have lots of data
    pmf[pmf == 0] = np.min(pmf[pmf != 0])
    filtered = pmf.copy()
    # plot filtered and unfiltered as lines using plotly, make a pd dataframe first
    px.line(pd.DataFrame({"unfiltered": unfiltered[0:4000], "filtered": filtered[0:4000]})).show()


    #pmf = np.maximum.accumulate(pmf[::-1])[::-1]  # will make sure a value is never smaller than the next, avoiding zeros
    #pmf = median_filter(pmf, size=50)

    #px.line(pmf[0:30000]).show()
    assert np.all(pmf > 0)

    # interpolate linearly to max genome size
    genome_size = np.sum(contig_sizes)
    logging.info("Genome size in bins is %d" % genome_size)
    remaining_dist = genome_size+1 - len(pmf)

    """
    first_value = pmf[-1]
    last_value = 0
    remaining_dist = genome_size+1 - len(pmf)
    rest = np.linspace(first_value, last_value, remaining_dist)
    rest[-1] = rest[-2]
    """

    # flat prob outside
    rest = np.zeros(remaining_dist) + pmf[-1] / 2

    pmf = np.concatenate([pmf, rest])
    pmf = pmf / np.sum(pmf)

    # set everything except beginning to be the same
    # we don't care about where read pairs are if they are far away, as we reach the backgrond noise
    pmf[10000:] = np.mean(pmf[10000:])

    assert np.all(pmf > 0)
    dist = DistanceDistribution.from_probabilities(pmf)
    #dist.smooth()
    return dist


def contigs_covering_percent_of_total(contig_sizes, percent_covered=0.4):
    sorted = np.sort(list(contig_sizes))[::-1]
    cumsum = np.cumsum(sorted)
    total_size = cumsum[-1]
    cutoff = np.searchsorted(cumsum, int(total_size) * percent_covered, side="right")
    contigs_to_use = np.argsort(contig_sizes)[::-1][:cutoff]
    return contigs_to_use


class BackgroundMatrix:
    """
    Used to create and represent a background-matrix (from a sparse interaction matrix)
    Can be used to get the sum of values in a given sized inter-matrix on the diagonal
    """
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    @classmethod
    def from_sparse_interaction_matrix(cls, sparse_matrix: SparseInteractionMatrix):
        contig_sizes = sparse_matrix._global_offset._contig_n_bins
        contigs_to_use = contigs_covering_percent_of_total(contig_sizes, percent_covered=0.4)
        logging.info(f"Using contigs {contigs_to_use} to estimate background matrix")
        smallest = np.min(contig_sizes[contigs_to_use])

        # sample from smaller matrices than smallest contig so we can sample
        # multiple times from the same matrix
        size = smallest // 2
        size = min(10000, size)
        logging.info(f"Using size {size} to estimate background matrix")
        background = np.zeros((size, size))
        n_sampled = 0
        logging.info("Estimating background matrix")
        for contig in tqdm(contigs_to_use):
            sample_positions = np.linspace(size, contig_sizes[contig]-size, 4)
            contig_start = sparse_matrix._global_offset.contig_first_bin(contig)
            contig_end = sparse_matrix._global_offset.contig_last_bin(contig, inclusive=False)
            for position in sample_positions:
                position = int(position)
                start = contig_start + position
                assert start > 0
                assert contig_start + size < contig_end
                submatrix = sparse_matrix.sparse_matrix[start-size:start, start:start+size].toarray()
                background += submatrix
                n_sampled += 1

        background = background / n_sampled
        background = csr_matrix(background)
        return cls(background)

    def get_score(self, y_size, x_size):
        assert y_size <= self.matrix.shape[0]
        assert x_size <= self.matrix.shape[1]
        return np.sum(self.matrix[-y_size:, :x_size])




