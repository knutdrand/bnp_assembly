import pytest
from bionumpy.util.testing import assert_bnpdataclass_equal

from bnp_assembly.simulation.missing_masker import mask_missing
from bnp_assembly.simulation.paired_read_positions import PairedReadPositions

@pytest.fixture
def read_pairs():
    return PairedReadPositions.from_entry_tuples([
        ('0', 10, '1', 20),
        ('0', 20, '1', 30),
        ('1', 30, '1', 40)
        ])

@pytest.fixture
def missing_dict():
    return {'0': [(0, 15)], '1': [(35, 45)]}


def test_missing_masker(read_pairs, missing_dict):
    masked = mask_missing(read_pairs, missing_dict)
    truth = PairedReadPositions.from_entry_tuples([
        ('0', 20, '1', 30),
    ])
    assert_bnpdataclass_equal(masked, truth)
    assert len(masked) == 1
