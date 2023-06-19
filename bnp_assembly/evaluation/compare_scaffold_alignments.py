from functools import lru_cache

import typing as tp

import numpy as np

from ..agp import ScaffoldAlignments
from ..scaffolds import Scaffolds


class ScaffoldComparison:
    def __init__(self, estimated_alignments: ScaffoldAlignments, true_alignments: ScaffoldAlignments):
        self._estimated_alignments = estimated_alignments
        self._true_alignments = true_alignments
        self._true_scaffold = Scaffolds.from_scaffold_alignments(true_alignments)
        self._estimated_scaffold = Scaffolds.from_scaffold_alignments(estimated_alignments)

    @property
    @lru_cache(maxsize=None)
    def _estimated_edge_dict(self) -> tp.Dict[str, tp.List[int]]:
        pass

    def edge_precision(self) -> float:
        if len(self._estimated_scaffold.edges) == 0:
            return 0
        return len(self._true_scaffold.edges & self._estimated_scaffold.edges) / len(self._estimated_scaffold.edges)

    def edge_recall(self) -> float:
        print("Edges not found: ", self._true_scaffold.edges-self._estimated_scaffold.edges)
        return len(self._true_scaffold.edges & self._estimated_scaffold.edges) / len(self._true_scaffold.edges)

    def missing_edges(self) -> tp.Set[str]:
        return self._true_scaffold.edges - self._estimated_scaffold.edges

    def false_edges(self):
        return self._estimated_scaffold.edges - self._true_scaffold.edges

