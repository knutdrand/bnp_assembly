import dataclasses
import itertools
from typing import Dict

from bionumpy import Genome

from .contigs import ContigLengths, ContigDict
from .scaffolds import Scaffolds
from .pair_distribution import MultiContigPairDistribution, MultiContigSignalPairDistribution, \
    MultiContigNoisePairDistribution
from ..agp import ScaffoldMap
from ..datatypes import GenomicLocationPair, StreamedGenomicLocationPair
from ..heatmap import create_heatmap_figure
from ..input_data import FullInputData, NumericInputData
from .missing_data_distribution import MissingRegionsDistribution
from ..io import PairedReadStream


def generate_contigs_and_reads():
    contig_dict = ContigDict(10, ContigLengths(100)).sample()
    scaffold_simulator = Scaffolds(contig_dict, 2)
    scaffolds = scaffold_simulator.sample()
    pair_distribution = MultiContigPairDistribution(contig_dict, p=0.1)
    reads = pair_distribution.sample(1000)
    return {'scaffolds': scaffolds, 'reads': reads}


def generate_training_from_contig_sizes(scaffold_aligments, n_reads=1000):
    scaffold_map = ScaffoldMap(scaffold_aligments)
    scaffold_sizes = scaffold_map.scaffold_sizes
    contig_sizes = scaffold_map.contig_sizes
    signal_distribution = MultiContigSignalPairDistribution(scaffold_sizes, p=0.00001)
    noise_distribution = MultiContigNoisePairDistribution(scaffold_sizes)
    read_distribution = MultiContigPairDistribution(signal_distribution, noise_distribution, p_signal=0.1)
    scaffold_read_pairs = read_distribution.sample(n_reads)
    mapped = scaffold_map.mask_and_map_location_pairs(scaffold_read_pairs)
    genome = Genome.from_dict(contig_sizes)
    scaffold_genome = Genome.from_dict(scaffold_sizes)
    genomic_location_pair = GenomicLocationPair(*(genome.get_locations(l) for l in (mapped.a, mapped.b)))
    # scaffold_pair = GenomicLocationPair(*(scaffold_genome.get_locations(l) for l in (scaffold_read_pairs.a, scaffold_read_pairs.b)))
    # create_heatmap_figure(scaffold_aligments, 1000, scaffold_genome, scaffold_pair)[0].show()
    numeric_location_pair = genomic_location_pair.get_numeric_locations()
    # pair = StreamedGenomicLocationPair(genomic_location_pair for _ in itertools.count())
    paired_read_stream = PairedReadStream(numeric_location_pair for _ in itertools.count())
    input_data = FullInputData(genome, paired_read_stream)
    return input_data, scaffold_genome, mapped
