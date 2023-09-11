from typing import Union, Iterable

import numpy as np
from .. import plotting
import bionumpy as bnp
from ..location import LocationPair
from ..make_scaffold import get_numeric_contig_name_translation
from ..scaffolds import Scaffolds


class ScaffoldingDebugger:
    def __init__(self, estimated_scaffolds: Scaffolds, truth_scaffolds: Scaffolds,
                 contigs: bnp.Genome, read_stream: Iterable[Iterable[LocationPair]],  plotting_folder: str = "./"):
        self.estimated_scaffolds = estimated_scaffolds
        self.truth_scaffolds = truth_scaffolds
        self.contigs = contigs
        self._read_stream = read_stream

        plotting.register(debug=plotting.ResultFolder(plotting_folder))
        self.px = plotting.px(name="debug")


        _, numeric_to_name_translation = get_numeric_contig_name_translation(self.contigs)
        self.contig_name_translation = {val: key for key, val in numeric_to_name_translation.items()}

    def get_reads_for_contig(self, contig_name):
        numeric_contig_name = self.contig_name_translation[contig_name]
        reads = next(self._read_stream)
        out = []
        for chunk in reads:
            mask = ((chunk.location_a.contig_id == numeric_contig_name) |
                    (chunk.location_b.contig_id == numeric_contig_name))
            out.append(chunk.subset_with_mask(mask))

        return LocationPair(
            np.concatenate([chunk.location_a for chunk in out]),
            np.concatenate([chunk.location_b for chunk in out])
        )

    def get_reads_between_contigs(self, contig_a, contig_b):
        reads = next(self._read_stream)
        #reads =

    def debug_edge(self, contig_name_a, contig_name_b):
        reads_a = self.get_reads_for_contig(contig_name_a)
        reads_b = self.get_reads_for_contig(contig_name_b)

        # todo: Find which contig contiga and contigb are linked to
        # find reads for them
        # make heatmap around all pairs of contigs

        print(reads_a)


    def finish(self):
        self.px.write_report()
