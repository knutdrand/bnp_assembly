from bnp_assembly.io import get_read_pairs
import pytest
import bionumpy as bnp


@pytest.fixture
def genome():
    return bnp.Genome.from_file('example_data/hifiasm.hic.p_ctg.fa')


def test_read_pairs(genome):
    location_pairs = get_read_pairs(genome, 'example_data/hic.sorted_by_read_name.bam')
    assert len(location_pairs.location_a) == 481
