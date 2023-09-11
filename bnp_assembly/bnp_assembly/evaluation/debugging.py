from typing import Union, Iterable

import numpy as np
from .. import plotting
import bionumpy as bnp

from ..io import PairedReadStream
from ..location import LocationPair
from ..make_scaffold import get_numeric_contig_name_translation
from ..scaffolds import Scaffolds


class ScaffoldingDebugger:
    def __init__(self,
                 estimated_scaffolds: Scaffolds,
                 truth_scaffolds: Scaffolds,
                 contigs: bnp.Genome,
                 reads: PairedReadStream,
                 plotting_folder: str = "./"):
        self.estimated_scaffolds = estimated_scaffolds
        self.truth_scaffolds = truth_scaffolds
        self.contigs = contigs
        self._read_stream = reads

        plotting.register(debug=plotting.ResultFolder(plotting_folder))
        self.px = plotting.px(name="debug")

        contig_sizes, numeric_to_name_translation = get_numeric_contig_name_translation(self.contigs)
        self.contig_name_translation = {val: key for key, val in numeric_to_name_translation.items()}
        self.contig_sizes = contig_sizes

    def get_reads_for_contig(self, contig_name):
        numeric_contig_name = self.contig_name_translation[contig_name]
        reads = next(self._read_stream)

        return LocationPair.from_multiple_location_pairs([
            chunk.filter_on_contig(numeric_contig_name) for chunk in reads
        ])

    def get_reads_between_contigs(self, contig_a, contig_b):
        contig_a = self.contig_name_translation[contig_a]
        contig_b = self.contig_name_translation[contig_b]
        reads = next(self._read_stream)

        return LocationPair.from_multiple_location_pairs([
            chunk.filter_on_contig(contig_a).filter_on_contig(contig_b)
            for chunk in reads
        ])

    def make_heatmap_for_two_contigs(self, contig_a, contig_b, bin_size=1000):
        contig_a_id = self.contig_name_translation[contig_a]
        contig_b_id = self.contig_name_translation[contig_b]
        reads_between = self.get_reads_between_contigs(contig_a, contig_b)
        reads = self.get_reads_between_contigs(contig_a, contig_b)

        heatmap_size = self.contig_sizes[contig_a_id] + self.contig_sizes[contig_b_id]
        heatmap = np.zeros((heatmap_size // bin_size, heatmap_size // bin_size))
        for read_a, read_b in zip(reads_between.location_a, reads_between.location_b):
            pos_a = read_a.offset
            pos_b = read_b.offset
            if read_a.contig_id == contig_b_id:
                pos_a += self.contig_sizes[contig_a_id]
            if read_b.contig_id == contig_a_id:
                pos_b += self.contig_sizes[contig_a_id]

            heatmap[pos_a // bin_size, pos_b // bin_size] += 1
            heatmap[pos_b // bin_size, pos_a // bin_size] += 1

        fig = self.px.imshow(np.log2(heatmap + 1), title="Heatmap for " + contig_a + " and " + contig_b)
        fig.show()
        return fig

    def debug_edge(self, contig_name_a, contig_name_b):
        self.make_heatmap_for_two_contigs(contig_name_a, contig_name_b)




        # todo: Find which contig contiga and contigb are linked to
        # find reads for them
        # make heatmap around all pairs of contigs

    def finish(self):
        self.px.write_report()
