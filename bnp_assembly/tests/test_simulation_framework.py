from bnp_assembly.simulation.contigs import ContigLengths, ContigDict
from bnp_assembly.simulation.scaffolds import Scaffolds


def test_simulate_scaffolds():
    contig_lengths = ContigLengths(100)
    contig_dict = ContigDict(10, contig_lengths)
    for i in range(10):
        scaffolds = Scaffolds(contig_dict, 2).sample()
        assert sum(scaffold.n_contigs for scaffold in scaffolds) == 10


def test_all():
    test_simulate_scaffolds()
