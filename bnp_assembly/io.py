import numpy as np
from .location import LocationPair, Location
from .datatypes import GenomicLocationPair
import bionumpy as bnp

def get_read_pairs(genome: bnp.Genome, bam_file_name):
    reads = genome.read_intervals(bam_file_name, buffer_type=bnp.io.bam.BamIntervalBuffer)
    locations = reads.get_location('stop')
    locations = Location(locations.chromosome.raw(), locations.position)
    return LocationPair(locations[::2], locations[1::2])

def get_genomic_read_pairs(genome: bnp.Genome, bam_file_name):
    reads = genome.read_intervals(bam_file_name, buffer_type=bnp.io.bam.BamIntervalBuffer)
    assert len(reads) % 2 == 0, reads 
    locations = reads.get_location('stop')
    return GenomicLocationPair(locations[::2], locations[1::2])
