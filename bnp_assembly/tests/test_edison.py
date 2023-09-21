from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.edison.scaffold_distance import main, distance
from bnp_assembly.edison import get_scaffold_distance

from bnp_assembly.scaffolds import Scaffolds


def test_scaffold_distance_main():
    main('../example_data/athalia_rosea.agp', '../example_data/athalia_rosea.agp')


def test_scaffold_distance():
    agp = ScaffoldAlignments.from_agp('../example_data/athalia_rosea.agp')
    d = agp.to_dict()
    distance(d, d)


def test_get_scaffold_distance():
    agp = ScaffoldAlignments.from_agp('../example_data/athalia_rosea.agp')
    get_scaffold_distance(agp, agp)
