import dataclasses
import typing as tp

import bionumpy as bnp
import numpy as np
from bionumpy.genomic_data import GenomicSequence

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.contig_graph import ContigPath


@dataclasses.dataclass
class Paths:
    paths: tp.List[ContigPath]
    translation_dict: tp.Dict[int, str]
    padding: int = 200

    def get_agp(self, contig_dict: tp.Dict[str, int]):
        alignments = []
        for i, path in enumerate(self.paths):
            scaffold_name = self.get_scaffold_name(i, path)
            offset = 0
            for j, dn in enumerate(path.directed_nodes):
                (contig_id, is_reverse) = dn.node_id, dn.orientation == '-'
                contig_name = self.translation_dict[contig_id]
                length = contig_dict[contig_name]
                if j > 0:

                    length += self.padding
                alignments.append(
                    (scaffold_name, offset, offset + length,
                     contig_name, 0, length, "+" if not is_reverse else "-")
                )
                offset += length
        return ScaffoldAlignments.from_entry_tuples(alignments)

    def get_sequence_entries(self, sequence_dict: GenomicSequence):
        paths = self.paths
        out_names = []
        out_sequences = []

        for i, path in enumerate(paths):
            sequences = []
            scaffold_name = self.get_scaffold_name(i, path)
            offset = 0
            for j, dn in enumerate(path.directed_nodes):
                (contig_id, is_reverse) = dn.node_id, dn.orientation == '-'
                if j > 0:
                    # adding 200 Ns between contigs
                    sequences.append(bnp.as_encoded_array('N' * self.padding, bnp.encodings.ACGTnEncoding))
                seq = sequence_dict[self.translation_dict[contig_id]]
                if is_reverse:
                    seq = bnp.sequence.get_reverse_complement(seq)
                sequences.append(bnp.change_encoding(seq, bnp.encodings.ACGTnEncoding))
            out_names.append(scaffold_name)
            out_sequences.append(np.concatenate(sequences))
        return bnp.datatypes.SequenceEntry.from_entry_tuples(zip(out_names, out_sequences))

    @staticmethod
    def get_scaffold_name(i, path):
        return f'scaffold{i}_' + ':'.join(f'{dn.node_id}{dn.orientation}' for dn in path.directed_nodes)
