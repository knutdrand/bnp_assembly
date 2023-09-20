import dataclasses

from .contigs import ContigLengths, ContigDict
from .scaffolds import Scaffolds
from .pair_distribution import MultiContigPairDistribution
from .missing_data_distribution import MissingRegionsDistribution

def generate_contigs_and_reads():
    contig_dict = ContigDict(10, ContigLengths(100)).sample()
    scaffold_simulator = Scaffolds(contig_dict, 2)
    scaffolds = scaffold_simulator.sample()
    pair_distribution = MultiContigPairDistribution(contig_dict, p=0.1)
    reads = pair_distribution.sample(1000)
    return {'scaffolds': scaffolds, 'reads': reads}

