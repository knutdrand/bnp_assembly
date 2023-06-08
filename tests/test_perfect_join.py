import pytest

import bionumpy as bnp
from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from bnp_assembly.io import get_genomic_read_pairs
from bnp_assembly.make_scaffold import make_scaffold


@pytest.mark.parametrize("folder_name", ["../example_data/simulated_perfect_join"])
def test_perfect_join(folder_name):
    genome_file_name = folder_name + "/contigs.chrom.sizes"
    genome = bnp.Genome.from_file(genome_file_name)
    bam_file_name = folder_name + "/reads.bam"
    reads = get_genomic_read_pairs(genome, bam_file_name)
    scaffold = make_scaffold(genome, reads, distance_measure='forbes2', threshold=1000, window_size=2500)
    alignments = scaffold.to_scaffold_alignments(genome, 200)
    true_alignments = ScaffoldAlignments.from_agp(folder_name + "/truth.agp")
    comparison = ScaffoldComparison(alignments, true_alignments)
    assert comparison.edge_recall() == 1
