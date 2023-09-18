from typing import Dict

import numpy as np
from bionumpy.bnpdataclass import bnpdataclass

from bnp_assembly.location import Location, LocationPair
from bionumpy.datatypes import LocationEntry


@bnpdataclass
class PairedLocationEntry:
    a: LocationEntry
    b: LocationEntry


class SingleContigPairDistribution:
    def __init__(self, contig_length, p: float):
        self._contig_length = contig_length
        self._p = p

    def sample(self, n_samples=1):
        distance = np.minimum(np.random.geometric(self._p, size=n_samples), self._contig_length - 2)
        first = np.random.randint(0, self._contig_length - distance)
        second = first + distance
        assert np.all(second < self._contig_length), (second, self._contig_length)
        assert np.all(first < self._contig_length), (first, self._contig_length)
        direction = np.random.choice([True, False], size=n_samples)
        return np.where(direction, first, first + distance), np.where(direction, first + distance, first)


class MultiContigSignalPairDistribution:
    def __init__(self, contig_lengths: Dict[str, int], p: float):
        self._contig_lengths = contig_lengths
        self._p = p

    def sample(self, n) -> PairedLocationEntry:
        total_length = sum(self._contig_lengths.values())
        ps = np.array(list(self._contig_lengths.values())) / total_length
        n_reads_per_contig = np.random.multinomial(n, ps)
        pairs = []
        for i, name in enumerate(self._contig_lengths.keys()):
            local_n = n_reads_per_contig[i]
            positions_a, positions_b = SingleContigPairDistribution(self._contig_lengths[name], self._p).sample(local_n)
            a = LocationEntry([name]*len(positions_a), positions_a)
            b = LocationEntry([name]*len(positions_b), positions_b)
            assert len(positions_a)==local_n, (len(positions_a), local_n)
            assert len(a) == local_n, (a, local_n)
            pair = PairedLocationEntry(a, b)
            assert len(pair) == local_n, pair
            pairs.append(pair)

        return np.concatenate(pairs)


class LocationDistribution:
    def __init__(self, contig_lengths: Dict[str, int]):
        self._contig_lengths = contig_lengths

    def sample(self, n):
        length_array = np.array(list(self._contig_lengths.values()))
        chrom_names = list(self._contig_lengths.keys())
        total_length = sum(length_array)
        ps = np.array(list(self._contig_lengths.values())) / total_length
        chrom_ids = np.random.choice(list(range(len(self._contig_lengths))), size=n, p=ps)
        locations = np.random.randint(0, length_array[chrom_ids])
        return LocationEntry([chrom_names[i] for i in chrom_ids], locations)


class MultiContigNoisePairDistribution:
    def __init__(self, contig_lengths: Dict[str, int]):
        self._contig_lengths = contig_lengths

    def sample(self, n) -> PairedLocationEntry:
        location_dist = LocationDistribution(self._contig_lengths)
        locations_a = location_dist.sample(n)
        locations_b = location_dist.sample(n)
        return PairedLocationEntry(locations_a, locations_b)


class MultiContigPairDistribution:
    def __init__(self, signal_distribution: MultiContigSignalPairDistribution, noise_distribution: MultiContigNoisePairDistribution, p_signal: float):
        self._signal = signal_distribution
        self._noise = noise_distribution
        self._p_signal = p_signal

    def sample(self, n) -> PairedLocationEntry:
        n_signal = np.random.binomial(n, self._p_signal)
        location_pairs = np.concatenate([
            self._signal.sample(n_signal),
            self._noise.sample(n - n_signal)])
        return location_pairs[np.random.permutation(n)]

