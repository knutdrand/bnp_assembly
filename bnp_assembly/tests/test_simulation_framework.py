from bionumpy import Genome
import pytest
from bnp_assembly.agp import ScaffoldAlignments, ScaffoldMap
from bnp_assembly.datatypes import GenomicLocationPair
from bnp_assembly.edison import get_scaffold_distance
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from bnp_assembly.heatmap import create_heatmap_figure
from bnp_assembly.make_scaffold import make_scaffold
from bnp_assembly.simulation.contigs import ContigLengths, ContigDict
from bnp_assembly.simulation.framework import generate_training_from_contig_sizes
from bnp_assembly.simulation.scaffolds import Scaffolds

scaffold_assembly = ScaffoldAlignments.from_entry_tuples([
    ("s1", 0, 50, "c1", 0, 50, "+"),
    ("s1", 70, 100, "c2", 0, 30, "+"),
    ("s2", 20, 40, "c3", 0, 20, "+")])


def test_simulate_scaffolds():
    contig_lengths = ContigLengths(100)
    contig_dict = ContigDict(10, contig_lengths)
    for i in range(10):
        scaffolds = Scaffolds(contig_dict, 2).sample()
        assert sum(scaffold.n_contigs for scaffold in scaffolds) == 10


def test_all():
    test_simulate_scaffolds()


def test_generate_contigs_and_reads():
    pass


@pytest.mark.skip  # slow
def test_generate_training_from_contig_sizes():
    input_data, *_ = generate_training_from_contig_sizes(scaffold_assembly)
    make_scaffold(input_data,
                  window_size=2500,
                  distance_measure='forbes3',
                  threshold=0,
                  splitting_method='matrix',
                  bin_size=1000,
                  max_distance=1000000)


@pytest.mark.skip  # slow
def test_generate_training_from_contig_sizes_big():
    big_scaffold_assembly = ScaffoldAlignments.from_agp('../example_data/athalia_rosea.agp')
    input_data, sg, contig_positions = generate_training_from_contig_sizes(big_scaffold_assembly, n_reads=100000)
    scaffold = make_scaffold(input_data, window_size=2500,
                             distance_measure='forbes3',
                             threshold=0,
                             splitting_method='matrix',
                             bin_size=1000,
                             max_distance=10000)
    agp = scaffold.to_scaffold_alignments(input_data.contig_genome, 1)
    scaffold_map = ScaffoldMap(agp)
    genome = Genome.from_dict(scaffold_map.scaffold_sizes)
    scaffold_pair = GenomicLocationPair(*(
       genome.get_locations(scaffold_map.map_to_scaffold_locations(pos)) for pos in (contig_positions.a,
                                                                                     contig_positions.b)))
    create_heatmap_figure(agp, 1000, genome, scaffold_pair, big_scaffold_assembly)[0].show()
    comparison = ScaffoldComparison(agp, big_scaffold_assembly)
    get_scaffold_distance(big_scaffold_assembly, agp)
    print(f'edge_recall\t{comparison.edge_recall()}\n')
    print(f'edge_precision\t{comparison.edge_precision()}\n')
