from typing import Iterable, List
import numpy as np
from bionumpy.bnpdataclass import bnpdataclass
from dataclasses import dataclass

@bnpdataclass
class Location:
    contig_id: int
    offset: int

    @classmethod
    def from_genomic_location(cls, genomic_location):
        return cls(genomic_location.chromosome.raw(), genomic_location.position)



@dataclass
class LocationPair:
    location_a: Location
    location_b: Location

    def __iter__(self):
        return (LocationPair(*p) for p in zip(self.location_a, self.location_b))

    def subset_with_mask(self, mask):
        return LocationPair(self.location_a[mask], self.location_b[mask])

    def filter_on_contig(self, contig_id):
        """Returns a new LocationPair object where any read pair has the contig"""
        mask = (self.location_a.contig_id == contig_id) | (self.location_b.contig_id == contig_id)
        return self.subset_with_mask(mask)

    def filter_on_two_contigs(self, contig_a, contig_b):
        mask = ((self.location_a.contig_id == contig_a) | (self.location_a.contig_id == contig_b)) & \
               ((self.location_b.contig_id == contig_a) | (self.location_b.contig_id == contig_b))
        return self.subset_with_mask(mask)

    @classmethod
    def from_multiple_location_pairs(cls, location_pairs: List['LocationPair']):
        return LocationPair(
            np.concatenate([chunk.location_a for chunk in location_pairs]),
            np.concatenate([chunk.location_b for chunk in location_pairs])
        )