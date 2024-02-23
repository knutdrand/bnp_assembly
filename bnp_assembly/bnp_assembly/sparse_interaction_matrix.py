import logging
from typing import Dict, Union

import bionumpy as bnp
from bionumpy.genomic_data.global_offset import GlobalOffset
import numpy as np
from bnp_assembly.graph_objects import Edge
from bnp_assembly.interaction_matrix import InteractionMatrix
from bnp_assembly.io import PairedReadStream
from scipy.sparse import bsr_array, lil_matrix
from scipy.sparse import csr_matrix


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

    def contig_last_bin(self, contig_id):
        return self.from_local_coordinates(contig_id, self.contig_sizes[contig_id]-1)


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
    def __init__(self, data: np.ndarray, global_offset: Union[NumericGlobalOffset, BinnedNumericGlobalOffset]):
        self._data = data
        self._global_offset = global_offset

    @classmethod
    def empty(cls, global_offset: Union[NumericGlobalOffset, BinnedNumericGlobalOffset]):
        size = global_offset.total_size()
        matrix = lil_matrix((size, size), dtype=float)
        return cls(matrix, global_offset)

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

    def get_edge_interaction_matrix(self, edge: Edge) -> np.ndarray:
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
        y_bin_start = self._global_offset.from_local_coordinates(contig_id, y_start)

        # end bins will be exclusive, i.e. first bin where first position is at end
        x_bin_end = int(x_bin_start + self._global_offset.distance_to_n_bins(contig_id, x_size) + 1)
        y_bin_end = int(y_bin_start + self._global_offset.distance_to_n_bins(contig_id, y_size) + 1)

        assert y_bin_end-y_bin_start == x_bin_end-x_bin_start
        submatrix = self.get_submatrix(slice(y_bin_start, y_bin_end), slice(x_bin_start, x_bin_end))

        #logging.info(f"New contig size: {x_end-x_start}, new bin size: {bins[1]-bins[0]}. Y: {y_start}, {y_end}, {y_end-y_start}. Bins: {bins}")
        new_global_offset = BinnedNumericGlobalOffset(np.array([x_size]),
                                                      np.array([x_bin_end-x_bin_start]),
                                                      np.array([0]))
        return SparseInteractionMatrix(submatrix, new_global_offset)

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

