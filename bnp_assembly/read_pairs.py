import numpy as np
from .location import LocationPair
import bionumpy as bnp


@bnp.bnpdataclass.bnpdataclass
class Location:
    contig_id: str
    offset: int


def get_read_pairs(bam_file_name):
    reads = bnp.open(bam_file_name, buffer_type=bnp.io.bam.BamIntervalBuffer).read()
    locations = Location(reads.chromosome, np.where(reads.strand == '+',
                                                    reads.stop, reads.start))
    return LocationPair(locations[::2], locations[1::2])
                         
