import numpy as np
from .interaction_matrix import InteractionMatrix, SplitterMatrix2
from .location import LocationPair, Location
from bionumpy.genomic_data import GenomeContext


class InteractionMatrixFactory:
    def __init__(self, contig_dict, bin_size):
        self._contig_dict = contig_dict
        self._bin_size = bin_size
        self._bin_size_dict, self._bin_offset, self._n_bins = self._calculate_grid()

    def _calculate_grid(self):
        bin_size_dict = {}
        offset_dict = {}
        cur_offset = 0
        bin_size = self._bin_size
        for contig, size in self._contig_dict.items():
            offset_dict[contig] = cur_offset
            n_bins = (size+bin_size-1)//bin_size
            cur_offset += n_bins
            bin_size_dict[contig] = size/n_bins
        return bin_size_dict, offset_dict, cur_offset

    def get_bin(self, contig, offset):
        local_bin = int(np.floor(offset/self._bin_size_dict[int(contig)]))
        return self._bin_offset[int(contig)]+local_bin

    def create_from_location_pairs(self, location_pairs: LocationPair) -> SplitterMatrix2:
        a_bins, b_bins = ([self.get_bin(location.contig_id, location.offset) for location in locations]
                          for locations in (location_pairs.location_a, location_pairs.location_b))
        data = np.zeros((self._n_bins, self._n_bins))
        np.add.at(data, (a_bins, b_bins), 1)
        np.add.at(data, (b_bins, a_bins), 1)
        return SplitterMatrix2(data, GenomeContext.from_dict({str(contig_id): size
                                                              for contig_id, size in self._contig_dict.items()}),
                               self._bin_size)

    def get_edge_bin_ids(self):
        return [self.get_bin(contig_id, 0) for contig_id in self._contig_dict]
