import pytest
import bnp_assembly.plotting as plotting
import bionumpy as bnp
from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from bnp_assembly.io import get_genomic_read_pairs
from bnp_assembly.make_scaffold import make_scaffold


@pytest.mark.parametrize("folder_name", ["../example_data/simulated_perfect_join"])
def test_perfect_join(folder_name):
    plotting.register(joining=plotting.ResultFolder('./tmp-joining/'))
    plotting.register(splitting=plotting.ResultFolder('./tmp-splitting/'))
    genome_file_name = folder_name + "/contigs.chrom.sizes"
    genome = bnp.Genome.from_file(genome_file_name)
    bam_file_name = folder_name + "/reads.bam"
    reads = get_genomic_read_pairs(genome, bam_file_name)
    scaffold = make_scaffold(genome, reads, distance_measure='forbes3', threshold=0.5, window_size=2500, splitting_method='poisson')
    alignments = scaffold.to_scaffold_alignments(genome, 200)
    true_alignments = ScaffoldAlignments.from_agp(folder_name + "/truth.agp")
    for key, group in bnp.groupby(alignments, 'scaffold_id'):
        print('\t', key, len(group))
        print(list(group.contig_id))
    for key, group in bnp.groupby(true_alignments, 'scaffold_id'):
        print('\t', key, len(group))
        print(group.contig_id)

    comparison = ScaffoldComparison(alignments, true_alignments)
    #@print(alignments)
    #@print(true_alignments)
    assert comparison.edge_recall() == 1, comparison.missing_edges()
    assert comparison.edge_precision() == 1, comparison.false_edges()
