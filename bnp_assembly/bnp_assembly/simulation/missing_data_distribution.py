from collections import defaultdict
from typing import Dict, List

import bionumpy as bnp
import numpy as np

from bnp_assembly.simulation.distribution import Distribution


class MissingRegionsDistribution(Distribution):
    def __init__(self, contig_dict: Dict[str, int], prob_missing, mean_size):
        if isinstance(contig_dict, bnp.datatypes.SequenceEntry):
            self._contig_dict = {str(entry.name): len(entry.sequence) for entry in contig_dict}
        else:
            self._contig_dict = contig_dict
        self._prob_missing = prob_missing
        self._mean_size = mean_size

    def _missing_dict_to_bed(self, missing_dict: Dict[str, List[int]])->bnp.datatypes.Interval:
        bed_entries = [(name, start, stop) for name, missing_regions in
                       missing_dict.items() for start, stop in missing_regions]
        if len(bed_entries) == 0:
            return bnp.datatypes.Interval.empty()
        return bnp.datatypes.Interval.from_entry_tuples(bed_entries)

    def sample(self, shape=()) -> bnp.datatypes.Interval:
        assert shape == ()
        missing_dict = defaultdict(list)
        for contig, size in self._contig_dict.items():
            if np.random.choice([True, False], p=[self._prob_missing, 1 - self._prob_missing]):
                missing_dict[contig].append((0, self._mean_size))
            if np.random.choice([True, False], p=[self._prob_missing, 1 - self._prob_missing]):
                missing_dict[contig].append((size - self._mean_size, size))
        return self._missing_dict_to_bed(missing_dict)
