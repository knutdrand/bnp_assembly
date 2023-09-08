import numpy as np

from ..agp import ScaffoldAlignments
from .compare_scaffold_alignments import ScaffoldComparison
from .. import plotting
import bionumpy as bnp

from ..graph_objects import Edge
from ..io import get_genomic_read_pairs_as_stream
from ..location import LocationPair
from ..make_scaffold import get_numeric_contig_name_translation


class ScaffoldingDebugger:
    def __init__(self, estimated_scaffolds: ScaffoldAlignments, truth_scaffolds: ScaffoldAlignments,
                 contigs: bnp.Genome, reads_bam: str,  plotting_folder: str = "./"):
        self.estimated_scaffolds = estimated_scaffolds
        self.truth_scaffolds = truth_scaffolds
        self.contigs = contigs
        plotting.register(debug=plotting.ResultFolder(plotting_folder))
        self.px = plotting.px(name="debug")
        self.comparison = ScaffoldComparison(estimated_scaffolds, truth_scaffolds)
        self.missing_edges = self.comparison.missing_edges()
        self.bam = reads_bam
        _, numeric_to_name_translation = get_numeric_contig_name_translation(self.contigs)
        self.contig_name_translation = {val: key for key, val in numeric_to_name_translation.items()}

    def get_reads_for_contig(self, contig_name):
        print(self.contig_name_translation)
        numeric_name = self.contig_name_translation[contig_name]
        locations_pairs = get_genomic_read_pairs_as_stream(self.contigs, self.bam)
        numeric_locations = locations_pairs.get_numeric_locations()
        out = []
        for chunk in numeric_locations:
            mask = (chunk.location_a.contig_id == numeric_name) | (chunk.location_b.contig_id == numeric_name)
            out.append(chunk.subset_with_mask(mask))

        return LocationPair(
            np.concatenate([chunk.location_a for chunk in out]),
            np.concatenate([chunk.location_b for chunk in out])
        )

    def debug_edge(self, contig_name_a, contig_name_b):
        reads_a = self.get_reads_for_contig(contig_name_a)
        reads_b = self.get_reads_for_contig(contig_name_b)

        # todo: Find which contig contiga and contigb are linked to
        # find reads for them
        # make heatmap around all pairs of contigs

        print(reads_a)


    def finish(self):
        self.px.write_report()
