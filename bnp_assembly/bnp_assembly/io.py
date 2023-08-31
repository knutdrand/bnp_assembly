import numpy as np
from .location import LocationPair, Location
from .datatypes import GenomicLocationPair
import bionumpy as bnp


def get_read_pairs(genome: bnp.Genome, bam_file_name: str, mapq_threshold=10) -> LocationPair:
    alignments = bnp.open(bam_file_name).read()
    mask = alignments.mapq > mapq_threshold
    interval = bnp.alignments.alignment_to_interval(alignments)
    reads = genome.get_intervals(interval)
    assert len(reads.data) % 2 == 0, len(reads.data)
    locations = reads.get_location('stop')
    locations = Location(locations.chromosome.raw(), locations.position)
    mask = mask[::2] & mask[1::2]
    return LocationPair(locations[::2][mask], locations[1::2][mask])


def get_genomic_read_pairs(genome: bnp.Genome, bam_file_name, mapq_threshold=10) -> GenomicLocationPair:
    alignments = bnp.open(bam_file_name).read()
    mask = alignments.mapq > mapq_threshold
    interval = bnp.alignments.alignment_to_interval(alignments)
    reads = genome.get_intervals(interval)
    location = reads.get_location('stop')
    mask = mask[::2] & mask[1::2]
    return GenomicLocationPair(location[::2][mask], location[1::2][mask])
