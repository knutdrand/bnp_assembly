import pytest
from bionumpy.genomic_data.global_offset import GlobalOffset
from bnp_assembly.io import PairedReadStream
from bnp_assembly.location import LocationPair, Location
from bnp_assembly.sparse_interaction_matrix import SparseInteractionMatrix


@pytest.fixture
def read_pairs3():
    read_pairs = PairedReadStream.from_location_pair(
        LocationPair(
            Location.from_entry_tuples([
                (0, 5), (0, 25), (0, 3)
            ]),
            Location.from_entry_tuples([
                (1, 5), (1, 25), (0, 2)
            ])
        )
    )
    return read_pairs


def test_from_reads(read_pairs3):
    contig_sizes = {0: 30, 1: 40}
    matrix = SparseInteractionMatrix.from_reads(contig_sizes, read_pairs3, 10)

    print(matrix)
