import pytest
from bnp_assembly.simulation.contig_simulation import introduce_unmappable_regions_to_contigs
import bionumpy as bnp

@pytest.fixture
def contig_sequences():
    return bnp.datatypes.SequenceEntry.from_entry_tuples([
        ("contig1", "AAACCACACTTTACTACTATC"),
        ("contig2", "AAACCACACTTTACTACTATC"),
        ("contig3", "AAACCACACTTTACTACTATC"),
        ("contig4", "AAACCACACTTTACTACTATC"),
    ])


def test_introduce_unmappable_regions_to_contigs(contig_sequences):
    new = introduce_unmappable_regions_to_contigs(contig_sequences, 0.1, 3)
    print(new)

