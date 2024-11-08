import logging
import time
from cgitb import small
from typing import Dict, Union, Tuple, List, Literal
import random
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
from scipy.sparse import lil_matrix, coo_array, coo_matrix
from scipy.sparse import csr_matrix
import sklearn
import matplotlib.pyplot as plt
from .plotting import px as px_func
import bionumpy as bnp


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

    @property
    def contig_n_bins(self):
        return self._contig_n_bins

    @property
    def contig_offsets(self):
        return self._contig_bin_offset

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
            n_bins = max(1, extended_size // approx_bin_size)  # never less than one bin
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

    def get_contigs_from_bins(self, bins: np.ndarray):
        # fast, by using a mask instead of np.searchsorted
        #mask = bins[:, None] >= self._contig_bin_offset
        #mask = np.logical_and(mask, bins[:, None] < self._contig_bin_offset + self._contig_n_bins)
        #contig_ids = np.argmax(mask, axis=1)
        mask = np.zeros(np.sum(self.contig_n_bins), dtype=int)
        for contig in range(len(self.contig_n_bins)):
            mask[self.contig_offsets[contig]:self.contig_offsets[contig]+self.contig_n_bins[contig]] = contig
        contig_ids = mask[bins].astype(int)
        return contig_ids

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

    def get_new_from_nodes(self, nodes: List[DirectedNode]):
        new_contig_sizes = np.array([self.contig_sizes[node.node_id] for node in nodes])
        new_contig_n_bins = np.array([self.contig_n_bins[node.node_id] for node in nodes])
        new_contig_offsets = np.insert(np.cumsum(new_contig_n_bins), 0, 0)
        return BinnedNumericGlobalOffset(new_contig_sizes, new_contig_n_bins, new_contig_offsets)

    def get_new_by_merging_nodes(self, nodes: List[List[int]]):
        """Returns a new global offset where new nodes are merges of the nodes given in the list of lists"""
        new_contig_sizes = np.array([sum(self.contig_sizes[node] for node in nodes) for nodes in nodes])
        new_contig_n_bins = np.array([sum(self.contig_n_bins[node] for node in nodes) for nodes in nodes])
        new_contig_offsets = np.insert(np.cumsum(new_contig_n_bins), 0, 0)
        return BinnedNumericGlobalOffset(new_contig_sizes, new_contig_n_bins, new_contig_offsets)

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
        t0 = time.perf_counter()
        #if not allow_nonsymmetric:
        #    assert (data.T != data).nnz == 0, "Matrix must be symmetric"
        self._data = data
        self._global_offset = global_offset
        logging.info(f"Init SparseInteractionMatrix took {time.perf_counter()-t0:.2f}s")

    def set_dtype(self, dtype):
        self._data = self._data.astype(dtype)

    @property
    def global_offset(self):
        return self._global_offset

    @property
    def n_contigs(self):
        return len(self._global_offset.contig_sizes)

    @property
    def contig_n_bins(self):
        return self._global_offset._contig_n_bins

    @property
    def contig_sizes(self):
        return self._global_offset.contig_sizes

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
    def from_pairs(cls, global_offset, genome: bnp.Genome, pairs_file_name):
        size = global_offset.total_size()
        matrix = lil_matrix((size, size), dtype=float)
        n_processed = 0

        all_rows = []
        all_cols = []

        with bnp.open(pairs_file_name) as f:
            chromosome_encoding = genome.get_genome_context().encoding

            for chunk in f:
                logging.info(f"{n_processed} pairs processed")
                chrom1_numeric = chromosome_encoding.encode(chunk.chrom1).raw()
                chrom2_numeric = chromosome_encoding.encode(chunk.chrom2).raw()
                pos1 = chunk.pos1
                pos2 = chunk.pos2
                global_pos1 = global_offset.from_local_coordinates(chrom1_numeric, pos1)
                global_pos2 = global_offset.from_local_coordinates(chrom2_numeric, pos2)

                #indexes = global_pos1 * size + global_pos2
                #matrix.ravel()[:] += np.bincount(indexes, minlength=size*size)

                #indexes = global_pos2 * size + global_pos1
                #matrix.ravel()[:] += np.bincount(indexes, minlength=size*size)
                all_rows.append(global_pos1)
                all_cols.append(global_pos2)
                all_rows.append(global_pos2)
                all_cols.append(global_pos1)

                # todo: Can speed up with bincout
                #for a, b in zip(global_pos1, global_pos2):
                #    matrix[a, b] += 1
                #    matrix[b, a] += 1

                n_processed += len(chunk)

        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        matrix = coo_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(size, size)).tocsr()

        #matrix = matrix.tocsr()
        return cls(matrix, global_offset)

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

    def set_global_offset(self, global_offset):
        self._global_offset = global_offset

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

    def approx_global_bin_size(self):
        return sum(self._global_offset.contig_sizes) / self._data.shape[0]

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

        for contig, (start, end) in tqdm(contig_clips.items(), desc="Adjusting interaction matrix", total=len(contig_clips)):
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
        new_data = self._data[~bins_to_remove, :][:, ~bins_to_remove]
        new_contig_offsets = np.insert(np.cumsum(new_contig_n_bins), 0, 0)
        new_global_offset = BinnedNumericGlobalOffset(new_contig_sizes, new_contig_n_bins, new_contig_offsets)

        self._global_offset = new_global_offset
        self._data = new_data
        assert self._data.shape[0] == np.sum(~bins_to_remove), f"{self._data.shape}, {np.sum(~bins_to_remove)}"
        assert self._data.shape[0] == self._data.shape[1]
        logging.info("Done trimming")

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
        logging.info(f"Getting matrix for path")
        if len(contigs) == len(self.contig_sizes) and np.all([contig.node_id for contig in contigs] == np.arange(len(contigs))) and all([contig.orientation == '+' for contig in contigs]):
            # path is same as matrix, return as is
            return self._data if as_raw_matrix else self
        rows = []
        for contig1 in tqdm(contigs):
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

    def plot(self, xaxis_names=None, title="", show_contigs=True):
        return self.plot_submatrix(None, None, xaxis_names=xaxis_names, title=title, show_contigs=show_contigs)

    def plot_submatrix(self, from_contig: int, to_contig: int, xaxis_names=None, title="", show_contigs=True):
        if from_contig == None and to_contig == None:
            matrix = self.sparse_matrix
            xaxis_names = [str(c) for c in range(self.n_contigs)]
            offsets = [self._global_offset.contig_first_bin(c) for c in range(0, self.n_contigs)]
        else:
            start = self._global_offset.contig_first_bin(from_contig)
            end = self._global_offset.contig_last_bin(to_contig, inclusive=False)
            matrix = self._data[start:end, start:end]
            if xaxis_names is None:
                xaxis_names = [str(c) for c in range(from_contig, to_contig+1)]
                xaxis_names = [x.replace("contig", "") for x in xaxis_names]
            offsets = [self._global_offset.contig_first_bin(c) - start for c in range(from_contig, to_contig+1)]

        logging.info(f"Total matrix size: {matrix.shape}")

        buckets = max(500, min(2000, matrix.shape[0]//200))
        logging.info(f"Number of buckets: {buckets}")
        fig, ax = matspy.spy_to_mpl(matrix, buckets=buckets, figsize=10, shading='relative')
        plt.title(title)
        if show_contigs:
            plt.vlines(offsets, 0, matrix.shape[0], color='b')
            plt.hlines(offsets, 0, matrix.shape[0], color='b')
            ax.set_xticks(offsets)
            ax.set_xticklabels(xaxis_names)
        return fig, ax

    def to_lil_matrix(self):
        t0 = time.perf_counter()
        logging.info("Converting to lil")
        self._data = self._data.tolil()
        logging.info(f"Converting to lil took {time.perf_counter()-t0:.2f}s")

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

    def get_new_by_grouping_nodes(self, path: List[List[DirectedNode]]):
        new = self.get_matrix_for_path2(
            [node for nodes in path for node in nodes], as_raw_matrix=False)
        new.set_global_offset(self._global_offset.get_new_by_merging_nodes(
            [[n.node_id for n in nodes] for nodes in path]
        ))
        return new

    def average_bin_value_outside_contigs(self):
        n_bins = self.sparse_matrix.shape[0]**2
        total = np.sum(self.sparse_matrix)
        total_inside_contigs = sum(
            np.sum(self.contig_submatrix(contig)) for contig in range(self.n_contigs)
        )
        n_bins_inside_contigs = sum(
            self._global_offset._contig_n_bins[contig]**2 for contig in range(self.n_contigs)
        )
        return (total - total_inside_contigs) / (n_bins - n_bins_inside_contigs)

    def edge_score(self, index, minimum_assumed_chromosome_size: int=1000, background_matrix: 'BackgroundMatrix' = None,
                   return_matrices=False):
        """Returns an edge score for splitting at the given index.
        The scores is minimum of the score of a right window and left window.
        For one window, we never look further than the previous contig sizes
        (in case we end up outside a chromosome)
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

        prev_contig_size = self._global_offset._contig_n_bins[index-1]
        next_contig_size = self._global_offset._contig_n_bins[index]
        matrix1 = matrix[-prev_contig_size:, :]  # never go further back than the prev
        matrix2 = matrix[:, :next_contig_size]  # never go further than the next
        if return_matrices:
            return matrix1, matrix2

        demominator1 = (matrix1.shape[1]*matrix1.shape[0])
        demominator2 = (matrix2.shape[1]*matrix2.shape[0])
        if background_matrix is not None:
            demominator1 = background_matrix.get_score(matrix1.shape[0], matrix1.shape[1])
            demominator2 = background_matrix.get_score(matrix2.shape[0], matrix2.shape[1])
            if demominator1 == 0 or demominator2 == 0:
                matspy.spy(background_matrix.matrix)
                plt.show()
                logging.error(matrix1.shape)
                logging.error(matrix1.shape)
                logging.error(matrix2.shape)

            assert demominator1 > 0
            assert demominator2 > 0

        score1 = np.sum(matrix1) / demominator1
        score2 = np.sum(matrix2) / demominator2
        return max(score1, score2)

    def merge_edges(self, edges) -> 'SparseInteractionMatrix':
        """Returns a new matrix where the two nodes in each of the edges have been merged
        and placed at the beginning of the matrix"""
        new_path = []
        picked_nodes = set()
        contig_n_bins = []
        new_contig_sizes = []
        for edge in edges:
            new_path.append(edge[0])
            new_path.append(edge[1])
            picked_nodes.add(edge[0].node_id)
            picked_nodes.add(edge[1].node_id)
            n_bins = self.contig_n_bins[edge[0].node_id] + self.contig_n_bins[edge[1].node_id]
            contig_n_bins.append(n_bins)
            new_contig_sizes.append(self.contig_sizes[edge[0].node_id] + self.contig_sizes[edge[1].node_id])

        # add the rest of the old nodes
        for node in range(self.n_contigs):
            if node not in picked_nodes:
                new_path.append(DirectedNode(node, '+'))
                contig_n_bins.append(self.contig_n_bins[node])
                new_contig_sizes.append(self.contig_sizes[node])

        new_contig_offsets = np.insert(np.cumsum(contig_n_bins), 0, 0)
        new_global_offset = BinnedNumericGlobalOffset(
            np.array(new_contig_sizes),
            np.array(contig_n_bins),
            new_contig_offsets)
        new_matrix = self.get_matrix_for_path2(new_path, backend=scipy.sparse.coo_matrix)

        return SparseInteractionMatrix(new_matrix, new_global_offset)

    def get_matrix_for_path2(self, path: List[DirectedNode], as_raw_matrix=True, backend=scipy.sparse.csr_matrix):
        """Fast vectorized version that uses searchsorted and some tricks
        to change sparse coordinates"""
        # Find current contiga/b for all nonzero
        node_ids = [contig.node_id for contig in path]
        if len(set(node_ids)) != len(node_ids):
            logging.warning("Path contains duplicate nodes")
            duplicate_node_ids = [node_id for node_id in node_ids if node_ids.count(node_id) > 1]
            logging.warning(f"Duplicate node ids: {duplicate_node_ids}")

        logging.info("Getting matix for path")

        t0 = time.perf_counter()
        rows, cols = np.nonzero(self.sparse_matrix)
        if isinstance(self.sparse_matrix, scipy.sparse.lil_matrix):
            data = self.sparse_matrix[rows, cols].toarray().ravel()
        elif isinstance(self.sparse_matrix, coo_matrix):
            data = self.sparse_matrix.data
        else:
            data = np.array(self.sparse_matrix[rows, cols]).ravel()

        logging.info(f"Time to get data from matrix: {time.perf_counter()-t0}")
        boundaries = self._global_offset._contig_bin_offset
        #contig_a = np.searchsorted(boundaries, rows, side='right')-1
        #contig_b = np.searchsorted(boundaries, cols, side='right')-1
        contig_a = self._global_offset.get_contigs_from_bins(rows)
        contig_b = self._global_offset.get_contigs_from_bins(cols)
        logging.info(f"Time to get contigs: {time.perf_counter()-t0}")

        new_contig_sizes = np.array([self._global_offset.contig_sizes[contig.node_id] for contig in path])
        new_contig_n_bins = np.array([self._global_offset._contig_n_bins[contig.node_id] for contig in path])
        new_bin_offsets = np.insert(np.cumsum(new_contig_n_bins), 0, 0)
        new_global_offset = BinnedNumericGlobalOffset(new_contig_sizes, new_contig_n_bins, new_bin_offsets)

        # how much to move positions inside contigs
        global_position_diffs = []
        for contig in range(self.n_contigs):
            current_offset = boundaries[contig]
            pos_in_new_path = [i for i, c in enumerate(path) if c.node_id == contig][0]
            new_offset = new_bin_offsets[pos_in_new_path]
            global_position_diffs.append(new_offset - current_offset)

        global_position_diffs = np.array(global_position_diffs)
        new_rows = rows + global_position_diffs[contig_a]
        new_cols = cols + global_position_diffs[contig_b]


        # adjust for reversed contigs
        is_reversed = np.array([contig.orientation == '-' for contig in path])

        #contigs_a_mask = np.searchsorted(new_bin_offsets, new_rows, side='right')-1
        #contigs_b_mask = np.searchsorted(new_bin_offsets, new_cols, side='right')-1
        contigs_a_mask = new_global_offset.get_contigs_from_bins(new_rows)
        contigs_b_mask = new_global_offset.get_contigs_from_bins(new_cols)
        logging.info(f"Time to searchsorted reverse: {time.perf_counter()-t0}")

        reversed_mask_a = is_reversed[contigs_a_mask]
        reversed_mask_b = is_reversed[contigs_b_mask]
        contigs_where_reversed_a = contigs_a_mask[reversed_mask_a]
        contigs_where_reversed_b = contigs_b_mask[reversed_mask_b]
        new_rows[reversed_mask_a] = new_bin_offsets[contigs_where_reversed_a] + new_contig_n_bins[contigs_where_reversed_a] - new_rows[reversed_mask_a] + new_bin_offsets[contigs_where_reversed_a] - 1
        new_cols[reversed_mask_b] = new_bin_offsets[contigs_where_reversed_b] + new_contig_n_bins[contigs_where_reversed_b] - new_cols[reversed_mask_b] + new_bin_offsets[contigs_where_reversed_b] - 1
        logging.info(f"Time to getting new rows/cols: {time.perf_counter()-t0}")

        new_total_bins = np.sum(new_contig_n_bins)
        # do not need next matrix to be indexed/compressed,
        logging.info(f"Time to getting path matrix: {time.perf_counter()-t0}")

        t0 = time.perf_counter()
        new_matrix = backend((data, (new_rows, new_cols)), shape=(new_total_bins, new_total_bins))
        logging.info(f"Time to init sparse matrix: {time.perf_counter()-t0}")

        if as_raw_matrix:
            return new_matrix

        return SparseInteractionMatrix(new_matrix, new_global_offset)

    def set_contig_to_zero(self, contig):
        """ find all indexes belonging to contig, remove these and create a new matrix
        """
        matrix = self.sparse_matrix
        start = self._global_offset.contig_first_bin(contig)
        end = self._global_offset.contig_last_bin(contig, inclusive=False)
        pass


    def remove_interactions_from_contigs(self, contigs: List[int]) -> 'SparseInteractionMatrix':
        logging.info(f"Removing interactions from contigs {contigs}")
        t0 = time.perf_counter()
        matrix = self
        size = matrix.sparse_matrix.shape[0]
        to_remove = np.zeros(size, dtype=bool)

        cols, rows = matrix.sparse_matrix.nonzero()
        data = matrix.sparse_matrix.data

        for contig in contigs:
            start = matrix._global_offset.contig_first_bin(contig)
            end = matrix._global_offset.contig_last_bin(contig, inclusive=False)
            to_remove[start:end] = True

        mask = np.logical_not(to_remove[rows]) & np.logical_not(to_remove[cols])
        new_data = data[mask]
        new_rows = rows[mask]
        new_cols = cols[mask]
        new_matrix = csr_matrix((new_data, (new_rows, new_cols)), shape=(size, size))
        logging.info(f"Time to remove interactions: {time.perf_counter()-t0}")
        return SparseInteractionMatrix(new_matrix, matrix._global_offset)




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


def estimate_distance_pmf_from_sparse_matrix2(sparse_matrix: SparseInteractionMatrix, set_to_flat_after_n_bins=1000) -> DistanceDistribution:
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
    for cutoff in [50, 500, 5000]:
        px_func(name='main').line(
            pd.DataFrame({"unfiltered": unfiltered[0:cutoff], "filtered": filtered[0:cutoff]}),
            title=f'Distance pmf, first {cutoff} bins')


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
    logging.info(f"Setting pmf to flat after {set_to_flat_after_n_bins} bins")
    pmf[set_to_flat_after_n_bins:] = np.mean(pmf[set_to_flat_after_n_bins:])

    assert np.all(pmf > 0)
    dist = DistanceDistribution.from_probabilities(pmf)
    #dist.smooth()
    return dist


def contigs_covering_percent_of_total(contig_sizes, percent_covered=0.4):
    sorted = np.sort(list(contig_sizes))[::-1]
    cumsum = np.cumsum(sorted)
    total_size = cumsum[-1]
    cutoff = np.searchsorted(cumsum, int(total_size) * percent_covered, side="right")
    logging.info(f"Using {cutoff}, sorted contigs: {np.argsort(contig_sizes)[::-1]}. Original sizes: {contig_sizes}")
    contigs_to_use = np.argsort(contig_sizes)[::-1][:cutoff+1]
    return contigs_to_use


class BackgroundMatrixInter:
    def __init__(self, average_bin_value: float):
        self._average_bin_value = average_bin_value

    def matrix(self, size):
        return np.ones((size, size)) * self._average_bin_value

    @property
    def average_value(self):
        return self._average_bin_value

    @classmethod
    def from_sparse_interaction_matrix(cls, sparse_matrix: SparseInteractionMatrix):
        return cls(sparse_matrix.average_bin_value_outside_contigs())


class BackgroundMatrix:
    """
    Used to create and represent a background-matrix (from a sparse interaction matrix)
    Can be used to get the sum of values in a given sized inter-matrix on the diagonal
    """
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    @property
    def matrices(self):
        return self.matrix

    @classmethod
    def from_sparse_interaction_matrix(cls, sparse_matrix: SparseInteractionMatrix, create_stack=False, max_contigs=10, n_per_contig=20):
        """
        If create_stack is True, will keep all matrices as a 3d matrix, not take mean of them
        """
        contig_sizes = sparse_matrix._global_offset._contig_n_bins
        contigs_to_use = contigs_covering_percent_of_total(contig_sizes, percent_covered=0.5)
        contigs_to_use = contigs_to_use[:max_contigs]
        logging.info(f"Using contigs {contigs_to_use} to estimate background matrix")
        smallest = np.min(contig_sizes[contigs_to_use])

        # sample from smaller matrices than smallest contig so we can sample
        # multiple times from the same matrix
        size = smallest // 3  # divide by three to make sure we can sample from multiple positions
        size = min(4000, size)  # divide by three to make sure we can sample from multiple positions
        logging.info(f"Using size {size} to estimate background matrix")
        background = np.zeros((size, size))
        all_backgrounds = np.zeros((n_per_contig*len(contigs_to_use), size, size))

        n_sampled = 0
        logging.info("Estimating background matrix")
        for contig in tqdm(contigs_to_use):
            #logging.info(f"Sampling from contig {contig} with size {contig_sizes[contig]}")
            sample_positions = np.linspace(size, contig_sizes[contig]-size, n_per_contig)
            contig_start = sparse_matrix._global_offset.contig_first_bin(contig)
            contig_end = sparse_matrix._global_offset.contig_last_bin(contig, inclusive=False)
            #logging.info(f"Sampling positions: {sample_positions}")
            for position in sample_positions:
                position = int(position)
                start = contig_start + position
                assert start > 0
                assert contig_start + size < contig_end
                #logging.info(f"Sampling from {start-size} to {start} and {start} to {start+size}")
                submatrix = sparse_matrix.sparse_matrix[start-size:start, start:start+size].toarray()
                if create_stack:
                    all_backgrounds[n_sampled] = submatrix
                    #all_backgrounds.append(submatrix)
                else:
                    background += submatrix
                n_sampled += 1

        if create_stack:
            return cls(all_backgrounds)

        background = background / n_sampled
        background = csr_matrix(background)
        return cls(background)

    def get_score(self, y_size, x_size):
        assert y_size <= self.matrix.shape[0]
        assert x_size <= self.matrix.shape[1]
        return np.sum(self.matrix[-y_size:, :x_size])

    def to_file(self, filename):
        np.save(filename, self.matrix)

    @classmethod
    def from_file(cls, filename):
        matrix = np.load(filename)
        return cls(matrix)


class BackgroundInterMatrices:
    """
    Represents many sampled background-matrices from what is assumed areas between chromosomes,
    can be used to get a distribution of the sum
    inside a given sized background-matrix
    """
    def __init__(self, matrices: np.ndarray):
        self.matrices = matrices

    @property
    def maxdist(self):
        return self.matrices.shape[1]

    def get_sums(self, y_size, x_size):
        """Returns sums sorted ascending"""
        assert y_size <= self.matrices.shape[1]
        assert x_size <= self.matrices.shape[2]
        return np.sort(np.sum(self.matrices[:, :y_size, :x_size], axis=(1, 2)))

    def plot(self):
        means = np.mean(self.matrices, axis=0)
        px.imshow(means).show()

    def get_percentile(self, y_size, x_size, observed_value):
        sums = self.get_sums(y_size, x_size)
        index = np.searchsorted(sums, observed_value, side="left")
        return (len(sums) - index) / len(sums)

    def get_percentile2(self, y_size, x_size, observed_value):
        sums = self.get_sums(y_size, x_size)
        mean = np.mean(sums)
        std = np.std(sums)
        if mean == 0 and std == 0:
            return 1
        assert mean > 0, f"{y_size}, {x_size}, std: {std}, sums: {sums}"
        assert std > 0, f"{y_size}, {x_size}, std: {std}, sums: {sums}"
        perc = 1 - scipy.stats.norm.cdf(observed_value, loc=mean, scale=std)

        assert not np.isnan(perc) and perc >= 0, (mean, std, observed_value)
        return perc

    def logpdf(self, y_size, x_size, observed_value):
        sums = self.get_sums(y_size, x_size)
        mean = np.mean(sums)
        std = np.std(sums)
        return scipy.stats.norm.logpdf(observed_value, loc=mean, scale=std)

    def logcdf(self, y_size, x_size, observed_value):
        sums = self.get_sums(y_size, x_size)
        mean = np.mean(sums)
        std = np.std(sums)
        return scipy.stats.norm.logcdf(observed_value, loc=mean, scale=std)

    @classmethod
    def weak_intra_interactions(cls, interaction_matrix: SparseInteractionMatrix, max_bins=1000, n_samples=50):
        """
        Attempts to sample the weaker interactions inside chromosomes by sampling closer to the diagonal
        """
        total_size = interaction_matrix.sparse_matrix.shape[1]
        max_bins = min(total_size // 30, max_bins)
        size = max_bins
        distance_from_diagonal = min(10000, total_size // 20)
        lowest_x_start = distance_from_diagonal + size
        highest_x_start = total_size - distance_from_diagonal - size
        matrices = np.zeros((n_samples, size, size))
        for i in tqdm(range(n_samples)):
            xstart = random.randint(lowest_x_start, highest_x_start)
            ystart = xstart - distance_from_diagonal
            submatrix = interaction_matrix.sparse_matrix[ystart:ystart+size, xstart:xstart+size].toarray()
            assert submatrix.shape[0] == size and submatrix.shape[1] == size
            matrices[i] = submatrix
            print(f"Sampling from {ystart} to {ystart+size} and {xstart} to {xstart+size}. Dist: {abs(ystart-xstart)}")

        return cls(matrices)


    @classmethod
    def weak_intra_interactions2(cls, interaction_matrix: SparseInteractionMatrix, max_bins=1000, n_samples=50, type=Literal['weak', 'strong']):
        """
        Sample from the outside of the biggest contigs
        """
        matrices = [matrix.toarray() for matrix in sample_intra_matrices(interaction_matrix, max_bins, n_samples, type)]
        return cls(np.array(matrices))


    @classmethod
    def from_sparse_interaction_matrix(cls, interaction_matrix: SparseInteractionMatrix, assumed_largest_chromosome_ratio_of_genome=1/10, n_samples=50, max_bins=1000):
        """
        Samples outside what is assumed to be largest chromosomes reach
        """
        matrices = [matrix.toarray().astype(np.float32) for matrix in sample_inter_matrices(interaction_matrix, assumed_largest_chromosome_ratio_of_genome, n_samples, max_bins)]
        return cls(np.array(matrices))


class BackgroundInterMatricesSingleBin:
    """
    Computes prob of a sum by only storing information about single bin mean/variance
    """
    def __init__(self, single_bin_mean:float, single_bin_variance: float):
        self._single_bin_mean = single_bin_mean
        self._single_bin_variance = single_bin_variance
        self.maxdist = 1000000

    def get_percentile2(self, y_size, x_size, observed_value):
        mean = y_size * x_size * self._single_bin_mean
        std = np.sqrt(y_size * x_size * self._single_bin_variance)
        perc = 1 - scipy.stats.norm.cdf(observed_value, loc=mean, scale=std)
        return perc

    def logpdf(self, y_size, x_size, observed_value):
        mean = y_size * x_size * self._single_bin_mean
        std = np.sqrt(y_size * x_size * self._single_bin_variance)
        return scipy.stats.norm.logpdf(observed_value, loc=mean, scale=std)

    def logcdf(self, y_size, x_size, observed_value):
        mean = y_size * x_size * self._single_bin_mean
        std = np.sqrt(y_size * x_size * self._single_bin_variance)
        return scipy.stats.norm.logcdf(observed_value, loc=mean, scale=std)

    def cdf(self, y_size, x_size, observed_value):
        y_size = max(1, y_size)
        x_size = max(1, x_size)
        mean = y_size * x_size * self._single_bin_mean
        std = np.sqrt(y_size * x_size * self._single_bin_variance)
        cdf = scipy.stats.norm.cdf(observed_value, loc=mean, scale=std)

        assert not np.isnan(cdf) and cdf >= 0, (mean, std, observed_value, y_size, x_size)

        return cdf

    @classmethod
    def from_sparse_interactin_matrix(cls, interaction_matrix: SparseInteractionMatrix,
                                   assumed_largest_chromosome_ratio_of_genome=1 / 10, n_samples=1000000):
        distance_from_diagonal = int(interaction_matrix.sparse_matrix.shape[1] * assumed_largest_chromosome_ratio_of_genome)
        samples = []

        while len(samples) < n_samples:
            i = np.random.randint(0, interaction_matrix.sparse_matrix.shape[1])
            j = np.random.randint(0, interaction_matrix.sparse_matrix.shape[1])
            #if abs(i-j) < distance_from_diagonal:
            #    continue

            value = float(interaction_matrix.sparse_matrix[i, j])
            samples.append(value)

        px.histogram(samples).show()
        mean = np.mean(samples)
        variance = np.var(samples)
        assert mean > 0
        assert variance > 0
        logging.info(f"Mean: {mean}, variance: {variance}")
        return cls(mean, variance)


class BackgroundInterMatricesMultipleBinsBackend:
    """Stores multiple bins in the backend, computes probs by sampling from them"""
    def __init__(self, bin_values: np.ndarray):
        self._bin_values = bin_values

    @classmethod
    def from_sparse_interaction_matrix(cls, interaction_matrix: SparseInteractionMatrix,
                                       assumed_largest_chromosome_ratio_of_genome=1 / 10, n_samples=1000000):

        distance_from_diagonal = int(interaction_matrix.sparse_matrix.shape[1] * assumed_largest_chromosome_ratio_of_genome)
        samples = []

        while len(samples) < n_samples:
            i = np.random.randint(0, interaction_matrix.sparse_matrix.shape[1])
            j = np.random.randint(0, interaction_matrix.sparse_matrix.shape[1])

            if abs(i-j) < distance_from_diagonal:
                continue

            value = float(interaction_matrix.sparse_matrix[i, j])
            samples.append(value)

        return cls(np.array(samples))

    def logpdf(self, size, observed_value):
        n_values = len(self._bin_values)
        n_samples = 100
        indexes = np.random.randint(0, n_values, (n_samples, size))
        samples = self._bin_values[indexes].sum(axis=1)
        mean = np.mean(samples)
        std = np.std(samples)
        return scipy.stats.norm.logpdf(observed_value, loc=mean, scale=std)


def get_number_of_reads_between_all_contigs(sparse_matrix: SparseInteractionMatrix, set_diagonal_to_zero=True):
    logging.info("Getting number of reads between all contigs")

    contig_offsets = sparse_matrix._global_offset.contig_offsets
    rows, cols = np.nonzero(sparse_matrix.sparse_matrix)
    values = np.array(sparse_matrix.sparse_matrix[rows, cols]).ravel()

    logging.info("Doing searchsorted")
    contig1 = np.searchsorted(contig_offsets, rows, side="right") - 1
    contig2 = np.searchsorted(contig_offsets, cols, side="right") - 1

    n_contigs = sparse_matrix.n_contigs
    out = np.zeros(n_contigs*n_contigs)
    indexes = contig1*n_contigs + contig2
    out = np.bincount(indexes, weights=values, minlength=n_contigs*n_contigs).reshape(n_contigs, n_contigs)

    if set_diagonal_to_zero:
        np.fill_diagonal(out, 0)
    return out


def sample_intra_matrices(interaction_matrix, max_bins, n_samples, type: Literal["weak", "strong"]):

    biggest_contigs = np.argsort(interaction_matrix._global_offset._contig_n_bins)[::-1]
    chosen_contigs = biggest_contigs[:min(5, 1+len(biggest_contigs)//4)]
    logging.info(f"Chosen contigs: {chosen_contigs}. Sizes: {interaction_matrix.contig_n_bins[chosen_contigs]}")
    smallest_contig_size = interaction_matrix.contig_n_bins[chosen_contigs[-1]]
    logging.info(f"Sampling from contigs {chosen_contigs}")

    if type == "weak":
        size = min(max_bins, smallest_contig_size // 4)
        min_distance_from_diagonal = smallest_contig_size // 2
    else:
        min_distance_from_diagonal = 0
        size = min(max_bins, smallest_contig_size // 2)
    logging.info(f"Sampling from size {size}")
    logging.info(f"Min distance from diagonal: {min_distance_from_diagonal}")

    #matrices = np.zeros((n_samples, size, size))

    n_sampled = 0
    for contig in chosen_contigs:
        contig_start_bin = interaction_matrix._global_offset.contig_first_bin(contig)
        contig_end_bin = interaction_matrix._global_offset.contig_last_bin(contig, inclusive=False)
        lowest_x_start = contig_start_bin + min_distance_from_diagonal + size
        highest_x_start = contig_end_bin - size

        n_to_sample_from_contig = n_samples // len(chosen_contigs) + 1
        for sample in range(n_to_sample_from_contig):
            if n_sampled >= n_samples:
                break

            xstart = random.randint(lowest_x_start, highest_x_start)
            lowest_y_start = contig_start_bin
            highest_y_start = xstart - min_distance_from_diagonal - size
            ystart = random.randint(lowest_y_start, highest_y_start)
            #logging.info(f"Sampling {type} from {ystart}:{ystart+size} and {xstart}:{xstart+size} on contig {contig}")
            #ystart = xstart - min_distance_from_diagonal
            #logging.info(f"Sampling from {ystart} to {ystart+size} and {xstart} to {xstart+size} on contig {contig} with size {contig_end_bin - contig_start_bin}")
            submatrix = interaction_matrix.sparse_matrix[ystart:ystart+size, xstart:xstart+size]
            assert submatrix.shape[0] == size and submatrix.shape[1] == size
            #matrices[n_sampled] = submatrix
            n_sampled += 1
            yield submatrix




def sample_inter_matrices(interaction_matrix: SparseInteractionMatrix,
                        assumed_largest_chromosome_ratio_of_genome=1 / 10, n_samples=50, max_bins=1000,
                          keep_close=False, weight_func=None):
    """
    Samples outside what is assumed to be largest chromosomes reach
    """
    distance_from_diagonal = int(
        interaction_matrix.sparse_matrix.shape[1] * assumed_largest_chromosome_ratio_of_genome)
    #distance_from_diagonal = min(20000, distance_from_diagonal)

    logging.debug(f"Assumed smallest chromosome size: {distance_from_diagonal}")
    largest_contig = np.max(interaction_matrix._global_offset._contig_n_bins)
    # size = min(5000, interaction_matrix.sparse_matrix.shape[1] // 10)
    size = min(max_bins, largest_contig)  # many bins is not necessary, and leads to large memory usage
    logging.debug(f"Making background matrices of size {size}")
    lowest_x_start = distance_from_diagonal + size
    highest_x_start = interaction_matrix.sparse_matrix.shape[1] - distance_from_diagonal - size

    #matrices = np.zeros((n_samples, size, size))
    for i in range(n_samples):
        xstart = random.randint(lowest_x_start, highest_x_start)
        lowest_y_start = 0
        highest_y_start = xstart - size - distance_from_diagonal
        if keep_close:
            lowest_y_start = highest_y_start-1  # only sample from this diagonal
        assert highest_y_start >= 0
        ystart = random.randint(lowest_y_start, highest_y_start)
        assert abs((ystart + size) - xstart) >= distance_from_diagonal
        # logging.info(f"Sampling from {ystart} to {ystart+size} and {xstart} to {xstart+size}")

        submatrix = interaction_matrix.sparse_matrix[ystart:ystart + size, xstart:xstart + size]
        assert submatrix.shape[0] == size and submatrix.shape[1] == size
        yield submatrix.toarray()


def sample_with_fixed_distance_inside_big_contigs(interaction_matrix: SparseInteractionMatrix,
                                                  max_bins=1000, n_samples=50,
                             distance_type=Union[int, Literal['close', 'far']]):

    biggest_contigs = np.argsort(interaction_matrix._global_offset._contig_n_bins)[::-1]
    chosen_contigs = biggest_contigs[:min(3, 1 + len(biggest_contigs) // 4)]
    logging.info(f"Chosen contigs: {chosen_contigs}. Sizes: {interaction_matrix.contig_n_bins[chosen_contigs]}")
    # only keep contigs with size larger than 50 bins
    chosen_contigs = [contig for contig in chosen_contigs if interaction_matrix.contig_n_bins[contig] > 50]
    assert len(chosen_contigs) > 0, "Could not find contigs large enough to sample from"
    smallest_contig_size = interaction_matrix.contig_n_bins[chosen_contigs[-1]]
    logging.info(f"Sampling from contigs {chosen_contigs}")

    size = max_bins
    size = min(size, smallest_contig_size//4)

    if distance_type == "close":
        #distance_from_diagonal = min(4000, smallest_contig_size//4)
        distance_from_diagonal = 1  #[10, 50]  #smallest_contig_size//16
        logging.info(f"Sampling close with distance: {distance_from_diagonal}")
    elif distance_type == "closest":
        distance_from_diagonal = 1
        #logging.info(f"Sampling closest with distance {distance_from_diagonal}")
    elif distance_type == "outer_contig":
        distance_from_diagonal = max(int(smallest_contig_size * 1/2), smallest_contig_size-20000)
        distance_from_diagonal = int(smallest_contig_size * 1/2)
        #logging.info(f"Sampling outer contig with distance {distance_from_diagonal}")
    elif distance_type == "far":
        #distance_from_diagonal = min(10000, smallest_contig_size//2)
        distance_from_diagonal = int(smallest_contig_size * 2 / 3)
        distance_from_diagonal = min(smallest_contig_size - size*2, distance_from_diagonal)
        logging.info(f"Sampling far with distance {distance_from_diagonal}")
    elif distance_type == "close_closest":
        #distance_from_diagonal = [min(smallest_contig_size//16, 50), 10]
        distance_from_diagonal = [smallest_contig_size//8, smallest_contig_size // 4]
        #distance_from_diagonal = [10000, 20000]
        factor = interaction_matrix.sparse_matrix.shape[1] // 500
        factor = min(factor, smallest_contig_size // 3)
        factor = min(smallest_contig_size - factor*4, factor)
        distance_from_diagonal = [factor, factor*2]
        distance_from_diagonal = 50  #[50]
        logging.info(f"Sampling close_closest with distances randomly picked from {distance_from_diagonal}")
    else:
        assert False


    #assert distance_from_diagonal < smallest_contig_size
    logging.debug(f"Distance from diagonal: {distance_from_diagonal}")

    n_sampled = 0
    for contig in chosen_contigs:
        contig_start_bin = interaction_matrix._global_offset.contig_first_bin(contig)
        contig_end_bin = interaction_matrix._global_offset.contig_last_bin(contig, inclusive=False)


        n_to_sample_from_contig = n_samples // len(chosen_contigs) + 1
        for sample in range(n_to_sample_from_contig):
            if isinstance(distance_from_diagonal, list):
                d = distance_from_diagonal[sample % 2]
            else:
                d = distance_from_diagonal

            lowest_x_start = contig_start_bin + d + size
            highest_x_start = contig_end_bin - size
            if n_sampled >= n_samples:
                break

            xstart = random.randint(lowest_x_start, highest_x_start)
            ystart = xstart - d - size
            submatrix = interaction_matrix.sparse_matrix[ystart:ystart + size, xstart:xstart + size]
            assert submatrix.shape[0] == size and submatrix.shape[1] == size
            # matrices[n_sampled] = submatrix
            n_sampled += 1
            effective_distance = abs(ystart+size - xstart)
            #logging.info("Distance: %d, size: %d" % (effective_distance, size))
            #matspy.spy(submatrix[::-1,])
            yield submatrix.toarray()[::-1,]


def filter_low_mappability(matrix: SparseInteractionMatrix, min_bins_left=10) -> SparseInteractionMatrix:
    """
    Creates a new interaction matrix where areas with low mappability have been filtered out
    """
    logging.info("Filtering regions with low mappability")
    s = matrix.sparse_matrix
    row_sums = np.array(s.sum(axis=1)).ravel()
    col_sums = np.array(s.sum(axis=0)).ravel()
    total_mapped = row_sums + col_sums
    mean = np.mean(total_mapped)
    sd = np.std(total_mapped)
    #threshold = mean - 1*sd
    threshold = mean / 5
    logging.info(f"Mean: {mean}, sd: {sd}, threshold: {threshold}")

    to_remove_mask = total_mapped < threshold
    to_remove = np.where(to_remove_mask)[0]

    contig_ids = matrix._global_offset.get_contigs_from_bins(to_remove)
    n_removed_in_each_contig = np.bincount(contig_ids, minlength=matrix.n_contigs)

    # filter to_remove, only keep those where not too much is removed
    new_to_remove = []
    remove_interactions_from_contigs = []
    #m = matrix.sparse_matrix.to_lil()
    #matrix.to_lil_matrix()
    for contig, n_removed in enumerate(n_removed_in_each_contig):
        # if removing mot of a contig, remove all of it. We don't want a few bins with many interactions
        # left as these are likely noise
        #if n_removed >= matrix.contig_n_bins[contig] - 2:
        size = matrix.contig_n_bins[contig]
        if n_removed / size >= 0.4:
            # remove all of contig
            remove_interactions_from_contigs.append(contig)
            #contig_start_bin = matrix._global_offset.contig_first_bin(contig)
            #contig_end_bin = matrix._global_offset.contig_last_bin(contig, inclusive=False)
            #new_to_remove.extend(np.arange(contig_start_bin, contig_end_bin)[1:-1])
            #logging.info("Ignoring contig %d" % contig)

            """
            contig_start_bin = matrix._global_offset.contig_first_bin(contig)
            contig_end_bin = matrix._global_offset.contig_last_bin(contig, inclusive=False)
            matrix.sparse_matrix[contig_start_bin:contig_end_bin, :] = 0
            matrix.sparse_matrix[:, contig_start_bin:contig_end_bin] = 0

            #prev = to_remove[contig_ids == contig]
            #contig_indexes = np.where(contig_ids == contig)[0]
            #new_to_remove.extend(contig_indexes[1:-1])
            """
        else:
            new_to_remove.extend(to_remove[contig_ids == contig])

        indexes = to_remove[contig_ids == contig]
        indexes = indexes - matrix._global_offset.contig_first_bin(contig)
        logging.info(f"Contig {contig}: Removed {n_removed}/{size} bins.")

        #if matrix.contig_n_bins[contig] - n_removed >= min_bins_left:
        #    # only remove if at least some bins left
        #    new_to_remove.extend(to_remove[contig_ids == contig])
        #else:
        #    logging.info(f"Not removing {n_removed} bins from contig {contig} since only {matrix.contig_n_bins[contig] - n_removed} bins left")

    #matrix.to_csr_matrix()

    to_remove = np.array(new_to_remove, int)
    to_remove_mask = np.zeros_like(to_remove_mask)
    to_remove_mask[to_remove] = True

    contig_ids = matrix._global_offset.get_contigs_from_bins(to_remove)
    n_removed_in_each_contig = np.bincount(contig_ids, minlength=matrix.n_contigs)
    print(f"N bins removed in each contig: {n_removed_in_each_contig}")

    for contig, n_removed in enumerate(n_removed_in_each_contig):
        logging.info(f"Removed {n_removed}/{matrix.contig_n_bins[contig]} bins from contig {contig}")

    new_contig_n_bins = matrix.contig_n_bins - n_removed_in_each_contig

    new_contig_offsets = np.insert(np.cumsum(new_contig_n_bins), 0, 0)[:-1]
    new_contig_sizes = np.array([
        size - n_removed_in_each_contig[contig]*matrix.contig_bin_size(contig) for contig, size
        in enumerate(matrix.contig_sizes)
        ])
    new_global_offset = BinnedNumericGlobalOffset(new_contig_sizes, new_contig_n_bins, new_contig_offsets)
    new_data = s[~to_remove_mask, :][:, ~to_remove_mask]
    assert new_global_offset.contig_n_bins.sum() == new_data.shape[0]
    assert new_global_offset.contig_offsets[-1] + new_global_offset.contig_n_bins[-1] == new_data.shape[0], f"{new_global_offset.contig_offsets[-1]} + {new_global_offset.contig_n_bins[-1]} != {new_data.shape[0]}"

    new = SparseInteractionMatrix(new_data, new_global_offset)
    new = new.remove_interactions_from_contigs(remove_interactions_from_contigs)
    return new



def weight_adjust_interaction_matrix(matrix: SparseInteractionMatrix, max_distance, scale=2.0):
    logging.info("Weight adjusting")

    s = matrix.sparse_matrix
    rows, cols = np.nonzero(s)
    data = s.data

    distance_from_diag = np.abs(rows - cols)
    # let weight go from scale to 1.0 linearly within distance, and be 1.0 for the rest
    weights = np.maximum(1.0, 1 + scale - scale * distance_from_diag / max_distance)
    new_data = data * weights
    new_sparse_matrix = csr_matrix((new_data, (rows, cols)), shape=s.shape)
    logging.info("Done weight adjusting")
    return SparseInteractionMatrix(new_sparse_matrix, matrix._global_offset)
