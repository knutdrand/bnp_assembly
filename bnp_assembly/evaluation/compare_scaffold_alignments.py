from functools import lru_cache

import typing as tp

from ..agp import ScaffoldAlignments


class ScaffoldComparison:
    def __init__(self, estimated_alignments: ScaffoldAlignments, true_alignments: ScaffoldAlignments):
        self._estimated_alignments = estimated_alignments
        self._true_alignments = true_alignments

    @property
    @lru_cache(maxsize=None)
    def _estimated_edge_dict(self) -> tp.Dict[str, tp.List[int]]:
        pass

    def edge_recall(self) -> float:
        pass



