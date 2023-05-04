from bionumpy.bnpdataclass import bnpdataclass
from dataclasses import dataclass
import bionumpy as bnp


@dataclass
class GenomicLocationPair:
    a: bnp.genomic_data.GenomicLocation
    b: bnp.genomic_data.GenomicLocation
