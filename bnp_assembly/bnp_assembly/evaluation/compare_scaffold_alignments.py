from functools import lru_cache

import typing as tp

import numpy as np
from bnp_assembly.graph_objects import Edge

from ..agp import ScaffoldAlignments
from ..scaffolds import Scaffolds
import logging

logger = logging.getLogger(__name__)


class ScaffoldComparison:
    def __init__(self, estimated_alignments: ScaffoldAlignments, true_alignments: ScaffoldAlignments):
        self._estimated_alignments = estimated_alignments
        self._true_alignments = true_alignments
        self._true_scaffold = Scaffolds.from_scaffold_alignments(true_alignments)
        self._estimated_scaffold = Scaffolds.from_scaffold_alignments(estimated_alignments)
        self.print_description()

    def print_description(self):
        print("Estimated scaffold:")
        print(self._estimated_alignments.get_description())

        print("True scaffolds: ")
        print(self._true_alignments.get_description())

    @property
    @lru_cache(maxsize=None)
    def _estimated_edge_dict(self) -> tp.Dict[str, tp.List[int]]:
        pass

    def edge_precision(self) -> float:
        if len(self._estimated_scaffold.edges) == 0:
            return 0
        logger.info(f'False edges: {self.false_edges()}')
        return len(self._true_scaffold.edges & self._estimated_scaffold.edges) / len(self._estimated_scaffold.edges)

    def edge_recall(self) -> float:
        print('Edges not found:')
        for edge in self.missing_edges():
            print(edge)
        return len(self._true_scaffold.edges & self._estimated_scaffold.edges) / len(self._true_scaffold.edges)

    @staticmethod
    def edge_weight(edge: Edge, contig_sizes) -> float:
        node1_size = contig_sizes[edge.from_node_side.node_id]
        node2_size = contig_sizes[edge.to_node_side.node_id]
        #return min(node1_size, node2_size)
        return (node1_size + node2_size) / 2

    def weighted_edge_recall(self, contig_sizes: tp.Dict[str, int]) -> float:
        """Weighted by the size of the contigs in the edge. Computes weights for true positive edges and divides by weight for true edges"""
        true_positives = self._true_scaffold.edges & self._estimated_scaffold.edges
        positives = self._true_scaffold.edges
        return (sum(self.edge_weight(edge, contig_sizes) for edge in true_positives) /
                sum(self.edge_weight(edge, contig_sizes) for edge in positives))

    def weighted_edge_precision(self, contig_sizes: tp.Dict[str, int]) -> float:
        true_positives = self._true_scaffold.edges & self._estimated_scaffold.edges
        true_positives_and_false_positives = self._estimated_scaffold.edges
        return (sum(self.edge_weight(edge, contig_sizes) for edge in true_positives) /
                sum(self.edge_weight(edge, contig_sizes) for edge in true_positives_and_false_positives))

    def missing_edges(self) -> tp.Set[str]:
        return self._true_scaffold.edges - self._estimated_scaffold.edges

    def false_edges(self):
        return self._estimated_scaffold.edges - self._true_scaffold.edges
