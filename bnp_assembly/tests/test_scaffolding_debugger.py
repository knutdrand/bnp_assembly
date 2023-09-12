import bionumpy as bnp

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.evaluation.debugging import ScaffoldingDebugger
import pytest
import numpy as np

from bnp_assembly.graph_objects import NodeSide
from bnp_assembly.io import PairedReadStream
from bnp_assembly.scaffolds import Scaffolds


def test_integration():
    dir = "example_data/yeast_small/"
    genome = bnp.Genome.from_file(dir + "hifiasm.hic.p_ctg.fa")
    truth = Scaffolds.from_scaffold_alignments(ScaffoldAlignments.from_agp(dir + "hifiasm.hic.p_ctg.agp"))
    scaffolds = Scaffolds.from_scaffold_alignments(ScaffoldAlignments.from_agp(dir + "scaffolds.agp"))

    genome = bnp.Genome.from_file(dir + "hifiasm.hic.p_ctg.fa")
    reads = PairedReadStream.from_bam(genome,
                                      dir + "hifiasm.hic.p_ctg.sorted_by_read_name.bam",
                                      mapq_threshold=10)



    print(truth)
    print(scaffolds)

    debugger = ScaffoldingDebugger(scaffolds, truth, genome, reads, plotting_folder="debugging")
    #debugger.debug_wrong_edges()
    debugger.debug_edge(
        DirectedNode("contig6", "-"),
        DirectedNode("contig7", "-")
    )
