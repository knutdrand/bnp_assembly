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


@pytest.mark.parametrize('mapping_ratio', [0, 0.1, 0.5, 0.9, 1])
def test_introduce_unmappable_regions_to_contigs(contig_sequences, mapping_ratio):
    new = introduce_unmappable_regions_to_contigs(contig_sequences, 0.1, 3, mapping_ratio=mapping_ratio)
    print(new)



