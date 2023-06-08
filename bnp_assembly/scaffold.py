from functools import lru_cache

import more_itertools

from .agp import ScaffoldAlignments
from .graph_objects import NodeSide, Edge

from collections import defaultdict


class Scaffold:
    def __init__(self, scaffold_alignments: ScaffoldAlignments):
        self._scaffold_alignments = scaffold_alignments

    @lru_cache()
    def scaffold_size(self, scaffold_id: str) -> int:
        return max(alignment.scaffold_end for alignment in self._scaffold_alignments if
                   str(alignment.scaffold_id) == str(scaffold_id))

    @property
    @lru_cache(maxsize=None)
    def edges(self):
        edges = set()
        contigs = defaultdict(list)
        print(self._scaffold_alignments)
        for alignment in self._scaffold_alignments:
            contigs[str(alignment.scaffold_id)].append(alignment)
        for contig, alignments in contigs.items():
            alignments.sort(key=lambda x: x.contig_start)
            print(alignments)
            for a, b in more_itertools.pairwise(alignments):
                node_side_a = NodeSide(str(a.contig_id), 'r' if a.orientation == '+' else 'l')
                node_side_b = NodeSide(str(b.contig_id), 'l' if b.orientation == '+' else 'r')
                edge = Edge(node_side_a, node_side_b)
                edges.add(edge)
                edges.add(edge.reverse())
        return edges

    def has_edge(self, edge: Edge) -> bool:
        return edge in self.edges

    @classmethod
    def from_scaffold_alignments(cls, scaffold_alignments: ScaffoldAlignments):
        return cls(scaffold_alignments)
