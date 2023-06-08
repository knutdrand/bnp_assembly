import numpy as np
from .location import LocationPair, Location
from .datatypes import GenomicLocationPair
import bionumpy as bnp


def get_read_pairs(genome: bnp.Genome, bam_file_name: str, mapq_threshold=10) -> LocationPair:
    alignments = bnp.open(bam_file_name).read()
    print(alignments)
    mask = alignments.mapq > mapq_threshold
    interval = bnp.alignments.alignment_to_interval(alignments)
    print(interval)
    reads = genome.get_intervals(interval)
    # reads = genome.read_intervals(bam_file_name, buffer_type=bnp.io.bam.BamIntervalBuffer)
    # mask = reads.get_data_field('mapq') > 10
    assert len(reads.data) % 2 == 0, len(reads.data)
    locations = reads.get_location('stop')
    locations = Location(locations.chromosome.raw(), locations.position)
    mask = mask[::2] & mask[1::2]
    return LocationPair(locations[::2][mask], locations[1::2][mask])


def get_genomic_read_pairs(genome: bnp.Genome, bam_file_name):
    reads = genome.read_intervals(bam_file_name, buffer_type=bnp.io.bam.BamIntervalBuffer)
    assert len(reads) % 2 == 0, reads
    locations = reads.get_location('stop')
    return GenomicLocationPair(locations[::2], locations[1::2])
