from bionumpy import Genome

from bnp_assembly.io import PairedReadStream
import bionumpy as bnp
from bionumpy.bnpdataclass import bnpdataclass
import pytest


@bnpdataclass
class BamEntryMock:
    chromosome: str
    position: int
    mapq: int
    flag: int
    cigar_op: str
    cigar_length: int


@pytest.fixture
def bam_entry():
    return BamEntryMock.from_entry_tuples(
        [
            ("chr1", 10, 15, 129, "M", 150),
            ("chr1", 50, 15, 65, "M", 150),
            ("chr2", 10, 20, 129, "M", 150),
            ("chr2", 15, 20, 65, "M", 150)
        ]
    )

@pytest.fixture
def genome():
    return Genome.from_dict({'chr1': 40,
                             'chr2': 50,
                             'chr3': 30})


def test_from_bam(genome, bam_entry):
    #stream = PairedReadStream.from_bam_entry(genome, bam_entry, mapq_threshold=10)
    genome = bnp.Genome.from_file("../example_data/hifiasm.hic.p_ctg.fa")
    stream = PairedReadStream.from_bam(genome, "../example_data/hic.sorted_by_read_name.bam", mapq_threshold=10)
    element = next(stream)
    for chunk in element:
        print(chunk)

    print("Chunk 2")
    element = next(stream)
    for chunk in element:
        print(chunk)
