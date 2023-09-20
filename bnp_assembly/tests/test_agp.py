import pytest
from bionumpy import LocationEntry
from bionumpy.util.testing import assert_bnpdataclass_equal

from bnp_assembly.agp import ScaffoldAlignments, ScaffoldMap

def test_agp():
    pass


def test_map_location():
    scaffold_assembly = ScaffoldAlignments.from_entry_tuples([
        ("s1", 0, 50, "c1", 0, 50, "+"),
        ("s1", 70, 100, "c2", 0, 30, "+"),
        ("s2", 20, 40, "c3", 0, 20, "+")])
    scaffold_map = ScaffoldMap(scaffold_assembly)
    locations = LocationEntry.from_entry_tuples([
        ("s1", 0),
        ("s1", 49),
        ("s1", 50),
        ("s1", 70),
        ("s1", 100),
        ("s2", 19),
        ("s2", 20)])
    true_locations = LocationEntry.from_entry_tuples([
        ("c1", 0),
        ("c1", 49),
        ("c2", 0),
        ("c3", 0)])
    map_locations = scaffold_map.mask_and_map_locations(locations)
    assert_bnpdataclass_equal(map_locations, true_locations)

