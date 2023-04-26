import numpy as np
from .location import LocationPair, Location
import bionumpy as bnp


# @bnp.bnpdataclass.bnpdataclass
# class Location:
#     contig_id: str
#     offset: int


def get_read_pairs(genome: bnp.Genome, bam_file_name):
    reads = genome.read_intervals(bam_file_name, buffer_type=bnp.io.bam.BamIntervalBuffer)
    # reads = bnp.open(bam_file_name, buffer_type=bnp.io.bam.BamIntervalBuffer).read()
    locations = reads.get_location('stop')
    # locations = Location(reads.chromosome, np.where(reads.strand == '+',
    #                                                 reads.stop, reads.start))
    locations = Location(locations.chromosome.raw(), locations.position)
    return LocationPair(locations[::2], locations[1::2])
