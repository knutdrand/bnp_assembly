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
