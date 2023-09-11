import bionumpy as bnp
from functools import lru_cache
import typing as tp
import more_itertools
import numpy as np
from bionumpy import Genome
from bionumpy.datatypes import SequenceEntry

from .agp import ScaffoldAlignments
from .contig_graph import ContigPath, DirectedNode, ContigPathSides
from .graph_objects import NodeSide, Edge

from collections import defaultdict


class Scaffold:
    def __init__(self, path: tp.List[DirectedNode], name: str = None):
        self._path: tp.List[DirectedNode] = path
        self._name = name

    def __repr__(self):
        return f'{self.name}: {self._path}'

    @property
    def name(self):
        return self._name

    @classmethod
    def from_contig_path(cls, contig_path: ContigPathSides, translation_dict: tp.Dict[int, str], name: str = None):
        translated_path = [DirectedNode(translation_dict[dn.node_id], dn.orientation) for dn in
                           contig_path.directed_nodes]
        return cls(translated_path, name)

    def to_contig_path(self, translation_dict: tp.Dict[str, int]):
        reverse_dict = {v: k for k, v in translation_dict.items()}
        translated_path = [DirectedNode(reverse_dict[dn.node_id], dn.orientation) for dn in self._path]
        return ContigPath.from_directed_nodes(translated_path)

    @property
    @lru_cache(maxsize=None)
    def edges(self):
        edges = set()
        for dn_a, dn_b in more_itertools.pairwise(self._path):
            node_side_a = NodeSide(dn_a.node_id, 'r' if dn_a.orientation == '+' else 'l')
            node_side_b = NodeSide(dn_b.node_id, 'l' if dn_b.orientation == '+' else 'r')
            edge = Edge(node_side_a, node_side_b)
            edges.add(edge)
            edges.add(edge.reverse())
        return edges

    @classmethod
    def from_alignments(cls, alignments: tp.List[ScaffoldAlignments], name: str = None):
        directed_nodes = [DirectedNode(str(alignment.contig_id), alignment.orientation) for alignment in alignments]
        return cls(directed_nodes, name)

    def to_scaffold_alignments(self, genome: Genome, padding):
        offset = 0
        alignments = []
        for j, dn in enumerate(self._path):
            contig_name = dn.node_id
            length = genome.get_genome_context().chrom_sizes[contig_name]
            if j > 0:
                length = length + padding
            alignment = (self.name, offset, offset + length,
                         contig_name, 0, length, dn.orientation)

            alignments.append(alignment)
            offset += length
        return ScaffoldAlignments.from_entry_tuples(alignments)

    def to_sequence(self, sequence_dict, padding):
        sequences = []
        for j, dn in enumerate(self._path):
            (contig_id, is_reverse) = dn.node_id, dn.orientation == '-'
            if j > 0:
                # adding 200 Ns between contigs
                sequences.append(bnp.as_encoded_array('N' * padding, bnp.encodings.ACGTnEncoding))
            seq = sequence_dict[contig_id]
            if is_reverse:
                seq = bnp.sequence.get_reverse_complement(seq)
            sequences.append(bnp.change_encoding(seq, bnp.encodings.ACGTnEncoding))
        return np.concatenate(sequences)


class Scaffolds:
    def __init__(self, scaffolds: tp.List[Scaffold]):
        self._scaffolds = scaffolds

    def __iter__(self):
        return iter(self._scaffolds)

    def __repr__(self):
        return '\n'.join(repr(scaffold) for scaffold in self._scaffolds)

    @property
    @lru_cache(maxsize=None)
    def edges(self):
        return set(edge for scaffold in self._scaffolds for edge in scaffold.edges)

    def has_edge(self, edge: Edge) -> bool:
        return edge in self.edges

    def get_neighbour(self, node_side: NodeSide) -> tp.Optional[NodeSide]:
        pass
        """
        for scaffold in self._scaffolds:
            for dn_a, dn_b in more_itertools.pairwise(scaffold._path):
                if dn_a.node_id == node_side.node_id and dn_a.orientation == node_side.orientation:
                    return NodeSide(dn_b.node_id, dn_b.orientation)
                if dn_b.node_id == node_side.node_id and dn_b.orientation == node_side.orientation:
                    return NodeSide(dn_a.node_id, dn_a.orientation)
        return None
        """

    def to_scaffold_alignments(self, genome, padding=200):
        return np.concatenate([scaffold.to_scaffold_alignments(genome, padding) for scaffold in self._scaffolds])

    @classmethod
    def from_scaffold_alignments(cls, scaffold_alignments: ScaffoldAlignments):
        scaffold_dict = defaultdict(list)
        for alignment in scaffold_alignments:
            scaffold_dict[str(alignment.scaffold_id)].append(alignment)

        scaffolds = [Scaffold.from_alignments(alignments, name=scaffold_id) for scaffold_id, alignments in
                     scaffold_dict.items()]
        return cls(scaffolds)

    @classmethod
    def create_name(cls, i, path=None):
        return f'scaffold{i}'

    @classmethod
    def from_contig_paths(cls, contig_paths: tp.List[ContigPath], translation_dict: tp.Dict[int, str]):
        scaffolds = [Scaffold.from_contig_path(contig_path, translation_dict, cls.create_name(i, contig_path)) for
                     i, contig_path in enumerate(contig_paths)]
        return cls(scaffolds)

    def to_sequence_entries(self, sequence_dict: str, padding: int = 200):
        return SequenceEntry.from_entry_tuples([(scaffold.name, scaffold.to_sequence(sequence_dict, padding)) for scaffold in self._scaffolds])
