from bionumpy.bnpdataclass import bnpdataclass
from dataclasses import dataclass

@bnpdataclass
class Location:
    contig_id: int
    offset: int


@dataclass
class LocationPair:
    location_a: Location
    location_b: Location
