from typing import Iterable, Union

import numpy as np

from bnp_assembly.location import LocationPair, Location


class ClipMapper:
    def __init__(self, contig_clips):
        self._contig_clips = contig_clips
        self._offsets = np.array([contig_clips[i][0] for i in range(len(contig_clips))])
        self._ends = np.array([contig_clips[i][1] for i in range(len(contig_clips))])

    def adjust_position(self, location: Location):
        return Location(location.contig_id, location.offset-self._offsets[location.contig_id])

    def is_in_range(self, location: Location):
        return (location.offset >= self._offsets[location.contig_id]) & (location.offset < self._ends[location.contig_id])

    def _create_mask(self, location_pairs: LocationPair):
        return self.is_in_range(location_pairs.location_a) & self.is_in_range(location_pairs.location_b)

    def map_coordinates(self, read_pairs: LocationPair):
        mask = self._create_mask(read_pairs)
        adjusted_postions_a = self.adjust_position(read_pairs.location_a)[mask]
        adjusted_postions_b = self.adjust_position(read_pairs.location_b)[mask]
        return LocationPair(adjusted_postions_a, adjusted_postions_b)

    def map_maybe_stream(self, read_pair_stram: Union[LocationPair, Iterable[LocationPair]]):
        if isinstance(read_pair_stram, LocationPair):
            return self.map_coordinates(read_pair_stram)
        return (self.map_coordinates(read_pair) for read_pair in read_pair_stram)


