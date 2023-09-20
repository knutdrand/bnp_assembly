from bnp_assembly.agp import ScaffoldAlignments
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


def test_generate_training_from_contig_sizes():
    input_data = generate_training_from_contig_sizes(scaffold_assembly)
    make_scaffold(input_data,
                  window_size=2500,
                  distance_measure='forbes3',
                  threshold=0,
                  splitting_method='matrix',
                  bin_size=1000,
                  max_distance=1000000)


def test_generate_training_from_contig_sizes_big():
    big_scaffold_assembly = ScaffoldAlignments.from_agp('../example_data/athalia_rosea.agp')
    input_data = generate_training_from_contig_sizes(big_scaffold_assembly, n_reads=1000000)
    make_scaffold(input_data,
                  window_size=2500,
                  distance_measure='forbes3',
                  threshold=0,
                  splitting_method='matrix',
                  bin_size=1000,
                  max_distance=10000)
