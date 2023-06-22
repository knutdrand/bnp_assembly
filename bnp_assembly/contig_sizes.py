import numpy as np


class ContigSizes:
    def __init__(self, size_array: np.ndarray):
        self._size_array = size_array

    @classmethod
    def from_dict(cls, contig_dict):
        max_contig_id = max(contig_dict)
        size_array = np.zeros(max_contig_id + 1, dtype=np.int64)
        for contig_id, contig_size in contig_dict.items():
            size_array[contig_id] = contig_size
        return cls(size_array)

    def __getitem__(self, item):
        return self._size_array[item]

    def __len__(self):
        return len(self._size_array)

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        return item < len(self._size_array)

    def keys(self):
        return range(len(self._size_array))

    def values(self):
        return self._size_array

    def items(self):
        return enumerate(self._size_array)
