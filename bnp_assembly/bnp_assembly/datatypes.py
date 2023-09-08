from typing import Iterable

from bionumpy.bnpdataclass import bnpdataclass
from dataclasses import dataclass
import bionumpy as bnp

from bnp_assembly.location import Location, LocationPair


@dataclass
class GenomicLocationPair:
    a: bnp.genomic_data.GenomicLocation
    b: bnp.genomic_data.GenomicLocation

    def get_numeric_locations(self) -> LocationPair:
        return LocationPair(Location.from_genomic_location(self.a),
                            Location.from_genomic_location(self.b))


@dataclass
class StreamedGenomicLocationPair:
    stream: Iterable[GenomicLocationPair]

    def get_numeric_locations(self):
        return (p.get_numeric_locations() for p in self.stream)


