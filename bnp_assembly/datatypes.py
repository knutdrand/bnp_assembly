from bionumpy.bnpdataclass import bnpdataclass
from dataclasses import dataclass
import bionumpy as bnp

from bnp_assembly.location import Location, LocationPair


@dataclass
class GenomicLocationPair:
    a: bnp.genomic_data.GenomicLocation
    b: bnp.genomic_data.GenomicLocation

    def get_numeric_locations(self):
        return LocationPair(Location.from_genomic_location(self.a),
                            Location.from_genomic_location(self.b))
