import logging

import bionumpy as bnp
from bionumpy.genomic_data.global_offset import GlobalOffset
import numpy as np
from bnp_assembly.interaction_matrix import InteractionMatrix
from bnp_assembly.io import PairedReadStream
from scipy.sparse import bsr_array, lil_matrix
from scipy.sparse import csr_matrix


class NumericGlobalOffset(GlobalOffset):
    def __init__(self, contig_sizes):
        self._names = list(contig_sizes.keys())
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
        return self.get_offset(contig_ids) + local_offsets


class SparseInteractionMatrix(InteractionMatrix):
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
        logging.info("Convert to row sparse")

        return cls(matrix, global_offset, bin_size)

