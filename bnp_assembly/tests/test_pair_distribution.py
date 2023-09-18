import pytest

from bnp_assembly.simulation.pair_distribution import MultiContigSignalPairDistribution, SingleContigPairDistribution, \
    MultiContigNoisePairDistribution, MultiContigPairDistribution


@pytest.fixture
def dist():
    return MultiContigSignalPairDistribution({'chr1': 100, 'chr2': 200}, p=0.1)

@pytest.fixture
def dist_noise():
    return MultiContigNoisePairDistribution({'chr1': 100, 'chr2': 200})

@pytest.fixture
def dist_all(dist, dist_noise):
    return MultiContigPairDistribution(dist, dist_noise, p_signal=0.3)



@pytest.fixture
def simple_dist():
    return SingleContigPairDistribution(100, p=0.1)


def test_simple_sample(simple_dist):
    assert all(len(locs) == 100 for locs in simple_dist.sample(100))


def test_sample(dist):
    res = dist.sample(1000)
    assert len(res) == 1000


def test_sample_noise(dist_noise):
    res = dist_noise.sample(1000)
    assert len(res) == 1000

def test_sample_all(dist_all):
    res = dist_all.sample(1000)
    assert len(res) == 1000

