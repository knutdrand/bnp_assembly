import os
from itertools import count

import pytest
import bnp_assembly.plotting as plotting
import bionumpy as bnp
from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from bnp_assembly.input_data import FullInputData
from bnp_assembly.io import get_genomic_read_pairs, PairedReadStream
from bnp_assembly.make_scaffold import make_scaffold


@pytest.mark.parametrize("folder_name", ["../example_data/simulated_perfect_join"])
# (joining, splitting) method
@pytest.mark.parametrize("methods", [
    #("forbes3", "matrix"),
    ("dynamic_heatmap", "matrix"),
])
def test_perfect_join(folder_name, methods):
    plotting.register(dynamic_heatmaps=plotting.ResultFolder('testplots/dynamic_heatmaps'))
    plotting.register(joining=plotting.ResultFolder('testplots/joining'))
    #plotting.register(splitting=plotting.ResultFolder('./tmp-splitting/'))
    genome_file_name = folder_name + "/contigs.chrom.sizes"
    genome = bnp.Genome.from_file(genome_file_name)
    bam_file_name = folder_name + "/reads.bam"
    reads = get_genomic_read_pairs(genome, bam_file_name)

    input_data = FullInputData(genome, PairedReadStream(([reads.get_numeric_locations()] for _ in count())))
    scaffold = make_scaffold(input_data, distance_measure=methods[0], threshold=0.2,
                             window_size=2500, splitting_method=methods[1], n_bins_heatmap_scoring=2)
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
    criterion = 1.0
    if methods[0] == "dynamic_heatmap":
        criterion = 0.94

    print("Recall", comparison.edge_recall())
    print("Precision", comparison.edge_precision())

    assert comparison.edge_recall() >= criterion, comparison.missing_edges()
    assert comparison.edge_precision() >= criterion, comparison.false_edges()
