import itertools
from collections import defaultdict
from functools import lru_cache

import more_itertools

from ..graph_objects import NodeSide, Edge
import typing as tp

from ..agp import ScaffoldAlignments


class Scaffold:
    def __init__(self, scaffold_alignments: ScaffoldAlignments):
        self._scaffold_alignments = scaffold_alignments

    @lru_cache()
    def scaffold_size(self, scaffold_id: str) -> int:
        return max(alignment.scaffold_end for alignment in self._scaffold_alignments if str(alignment.scaffold_id) == str(scaffold_id))

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

    def _edges(self):
        edges = set()
        end_to_node_side_dict = {
            (str(alignment.scaffold_id), int(alignment.scaffold_end)): NodeSide(str(alignment.contig_id),
                                                                                       'r' if alignment.orientation == '+' else 'l')
            for alignment in self._scaffold_alignments}
        start_to_node_side_dict = {(str(alignment.scaffold_id), int(alignment.scaffold_start)): NodeSide(str(alignment.contig_id),
                                                                                                         'l' if str(alignment.orientation) == '+' else 'r')
                                   for alignment in self._scaffold_alignments}
        for alignment in self._scaffold_alignments:
            start_side = NodeSide(str(alignment.contig_id), 'l')
            end_side = NodeSide(str(alignment.contig_id), 'r')
            if alignment.orientation == '-':
                start_side, end_side = end_side, start_side
            if alignment.scaffold_start != 0:
                start_match = end_to_node_side_dict[(str(alignment.scaffold_id), int(alignment.scaffold_start))]
                edges.add(Edge(start_side, start_match))
            if alignment.scaffold_end != self.scaffold_size(str(alignment.scaffold_id)):
                end_match = start_to_node_side_dict[(str(alignment.scaffold_id), int(alignment.scaffold_end))]
                edges.add(Edge(end_side, end_match))

        return edges

    def has_edge(self, edge: Edge) -> bool:
        return edge in self.edges

    @classmethod
    def from_scaffold_alignments(cls, scaffold_alignments: ScaffoldAlignments):
        return cls(scaffold_alignments)


class ScaffoldComparison:
    def __init__(self, estimated_alignments: ScaffoldAlignments, true_alignments: ScaffoldAlignments):
        self._estimated_alignments = estimated_alignments
        self._true_alignments = true_alignments
        self._true_scaffold = Scaffold.from_scaffold_alignments(true_alignments)
        self._estimated_scaffold = Scaffold.from_scaffold_alignments(estimated_alignments)

    @property
    @lru_cache(maxsize=None)
    def _estimated_edge_dict(self) -> tp.Dict[str, tp.List[int]]:
        pass

    def edge_recall(self) -> float:
        print(self._true_scaffold.edges)
        return len(self._true_scaffold.edges & self._estimated_scaffold.edges) / len(self._true_scaffold.edges)
