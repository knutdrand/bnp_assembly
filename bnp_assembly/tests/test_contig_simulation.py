from bnp_assembly.simulation.contig_simulation import simulate_contigs_from_genome, \
    random_locations_with_some_short_intervals
import pytest
from bionumpy.datatypes import SequenceEntry
import numpy as np


@pytest.fixture
def genome():
    return SequenceEntry.from_entry_tuples([
        ["chr1", "ACTGACTGACTG"],
        ["chr2", "GGGGGGGGGGGGGGGGGGG"]
    ])


def test_simulate_contigs(genome):
    rng = np.random.default_rng(1)
    simulated = simulate_contigs_from_genome(genome, 4, min_contig_size=3, rng=rng)

    assert len(simulated.contigs) == 4
    assert np.all(simulated.contigs.sequence.shape[1] >= 3)

    print(simulated.alignment)
    alignment = simulated.alignment
    assert np.all(alignment.scaffold_end - alignment.scaffold_start == alignment.contig_end - alignment.contig_start)


def test_random_locations_with_some_short_intervals():

    locations = random_locations_with_some_short_intervals(0, 100, 10, ratio_small=0.3, small_size=4)
    print(locations)

