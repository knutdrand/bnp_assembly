import dataclasses
from typing import Union, Iterable, Dict

from bionumpy import Genome

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.datatypes import GenomicLocationPair, StreamedGenomicLocationPair
from bnp_assembly.io import PairedReadStream
from bnp_assembly.location import LocationPair


@dataclasses.dataclass
class FullInputData:
    contig_genome: Genome
    paired_read_stream: Union[PairedReadStream, Iterable[Union[GenomicLocationPair, StreamedGenomicLocationPair]]]



@dataclasses.dataclass
class FullInputAndTruth:
    input_data: FullInputData
    truth: ScaffoldAlignments


@dataclasses.dataclass
class NumericInputData:
    contig_dict: Dict[int, int]
    location_pairs: Union[Iterable[LocationPair], PairedReadStream]

