# Naive simulation of actualy HiC reads
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from bnp_assembly.simulation.distribution import Distribution
from bnp_assembly.simulation.missing_data_distribution import MissingRegionsDistribution
from bnp_assembly.simulation.missing_masker import mask_missing
from bnp_assembly.simulation.paired_read_positions import PairedReadPositions

logging.basicConfig(level=logging.INFO)
import bionumpy as bnp
import numpy as np
import typer

def _random_genome_locations(contigs, n_reads, read_length):
    genome_size = np.sum(contigs.sequence.shape[1])
    contig_weights = contigs.sequence.shape[1] / genome_size
    contig_lengths = contigs.sequence.shape[1]

    drawn_contigs = np.random.choice(np.arange(len(contigs.sequence)), size=n_reads, p=contig_weights)
    drawn_positions = np.random.randint(0, contig_lengths[drawn_contigs] - read_length)
    return drawn_positions, drawn_contigs


def simulate(contigs: bnp.datatypes.SequenceEntry, n_reads: int, read_length: int, fragment_size_mean: int,
             signal: float, read_name_prefix: str = '') -> List[bnp.datatypes.SequenceEntryWithQuality]:
    read_pair_dist = PairedReadPositionsDistribution(contigs, fragment_size_mean, read_length, signal)
    paired_reads = read_pair_dist.sample(n_reads)
    return get_reads_from_positions(contigs, paired_reads, read_length, read_name_prefix)


def get_reads_from_positions(contigs, paired_reads, read_length, read_name_prefix):
    single_reads = []
    for i, (contig_names, positions) in enumerate(
        [[paired_reads.contig_1, paired_reads.position_1], [paired_reads.contig_2, paired_reads.position_2]]):
        out_sequences = []
        logging.info(f"Writing {len(contig_names)} reads contig number {i}")
        for read_number, (contig, position) in enumerate(zip(contig_names, positions)):
            sequence = contigs.sequence[contig][position:position + read_length]
            out_sequences.append((f"{read_name_prefix}{read_number}", sequence, "I" * len(sequence)))

        single_reads.append(bnp.datatypes.SequenceEntryWithQuality.from_entry_tuples(out_sequences))
    return single_reads


def simulate_hic_from_file(contigs_file_name: str, n_reads: int, read_length: int, fragment_size_mean: int,
                           signal: float,
                           out_base_name: str,
                           read_name_prefix: str, do_mask_missing: bool = False):
    contigs = bnp.open(contigs_file_name).read()
    paired_reads_dist = PairedReadPositionsDistribution(contigs, fragment_size_mean, read_length, signal)
    if do_mask_missing:
        missing_mask = MissingRegionsDistribution(contigs, 0.1, 10000)
    else:
        missing_mask = None
    reads_sequence_dist = ReadSimulator(contigs, paired_reads_dist, read_length, read_name_prefix, missing_mask)
    data_stream = reads_sequence_dist.sample(n_reads)

    # data_stream = simulate(contigs_file_name, n_reads, read_length, fragment_size_mean, signal, read_name_prefix)

    for i, data in enumerate(data_stream):
        with bnp.open(out_base_name + f"{i + 1}.fq.gz", "w") as f:
            f.write(data)


class _MissingRegionsDistribution(Distribution):
    def __init__(self, contig_dict: Dict[str, int], prob_missing, mean_size):
        if isinstance(contig_dict, bnp.datatypes.SequenceEntry):
            self._contig_dict = {str(entry.name): len(entry.sequence) for entry in contig_dict}
        else:
            self._contig_dict = contig_dict
        self._prob_missing = prob_missing
        self._mean_size = mean_size

    def _missing_dict_to_bed(self, missing_dict: Dict[str, List[int]])->bnp.datatypes.Interval:
        bed_entries = [(name, start, stop) for name, missing_regions in
                       missing_dict.items() for start, stop in missing_regions]
        if len(bed_entries) == 0:
            return bnp.datatypes.Interval.empty()
        return bnp.datatypes.Interval.from_entry_tuples(bed_entries)

    def sample(self, shape=()) -> bnp.datatypes.Interval:
        assert shape == ()
        missing_dict = defaultdict(list)
        for contig, size in self._contig_dict.items():
            if np.random.choice([True, False], p=[self._prob_missing, 1 - self._prob_missing]):
                missing_dict[contig].append((0, self._mean_size))
            if np.random.choice([True, False], p=[self._prob_missing, 1 - self._prob_missing]):
                missing_dict[contig].append((size - self._mean_size, size))

        return self._missing_dict_to_bed(missing_dict)


class PairedReadPositionsDistribution(Distribution):
    def __init__(self, contigs, fragment_size_mean: float, read_length: int, signal: float):
        self._contigs = contigs
        self._fragment_size_mean = fragment_size_mean
        self._read_length = read_length
        self._signal = signal

    def sample(self, n):
        # Make sure we get exactly n reads
        t = 0
        buffers = []
        while t < n:
            buffer = self._sample(n)
            buffers.append(buffer)
            t += len(buffer)
        return np.concatenate(buffers)[:n]

    def _sample(self, n):
        if isinstance(self._contigs, Distribution):
            self._contigs = self._contigs.sample()
        all_contigs1, all_contigs2, all_positions1, all_positions2, _ = simulate_raw(self._contigs,
                                                                                     self._fragment_size_mean, n,
                                                                                     self._read_length, self._signal)
        return PairedReadPositions(all_contigs1, all_positions1, all_contigs2, all_positions2)


class ReadSimulator(Distribution):
    def __init__(self, contigs: bnp.datatypes.SequenceEntry, pair_read_positions: PairedReadPositionsDistribution,
                 read_length: int, read_name_prefix: str = '',
                 missing_mask: Optional[MissingRegionsDistribution] = None):
        self._contigs = contigs
        self._pair_read_positions = pair_read_positions
        self._read_length = read_length
        self._read_name_prefix = read_name_prefix
        self._missing_mask = missing_mask

    def sample(self, n_reads):
        pair_read_positions = self._pair_read_positions.sample(n_reads)
        if self._missing_mask is not None:
            missing_regions = self._missing_mask.sample()
            pair_read_positions = mask_missing(pair_read_positions, missing_regions)
        return get_reads_from_positions(self._contigs, pair_read_positions, self._read_length, self._read_name_prefix)


def simulate_raw(contigs: bnp.datatypes.SequenceEntry, fragment_size_mean: float, n_reads: int, read_length: int,
                 signal: float):
    n_reads_signal = int(n_reads * signal)
    n_reads_noise = n_reads - n_reads_signal
    logging.info(f"Will simulate {n_reads_signal} reads with signal and {n_reads_noise} noise reads")
    contig_sizes = contigs.sequence.shape[1]
    all_positions1 = []
    all_contigs1 = []
    all_positions2 = []
    all_contigs2 = []

    for read_type, n_reads in [("signal", n_reads_signal), ("noise", n_reads_noise)]:
        # draw half of reads at random positions
        positions1, contigs1 = _random_genome_locations(contigs, n_reads // 2, read_length)

        if read_type == "signal":
            # draw other half at an insert size away from the first half with the same contig id
            insert_sizes = np.random.geometric(1 / fragment_size_mean, n_reads // 2)
            logging.info("Using insert sizes: %s" % insert_sizes)
            positions2 = positions1 + insert_sizes * np.random.choice([1, -1], size=len(insert_sizes))  # right bias
            contigs2 = contigs1.copy()

        elif read_type == "noise":
            # draw other half at random locations
            positions2, contigs2 = _random_genome_locations(contigs, n_reads // 2, read_length)

        all_contigs1.append(contigs1)
        all_contigs2.append(contigs2)
        all_positions1.append(positions1)
        all_positions2.append(positions2)
    all_contigs1 = np.concatenate(all_contigs1)
    all_positions1 = np.concatenate(all_positions1)
    all_contigs2 = np.concatenate(all_contigs2)
    all_positions2 = np.concatenate(all_positions2)
    # mask out read pairs where one read is outside of the contig
    mask = (all_positions1 < contig_sizes[all_contigs1] - read_length) & \
           (all_positions2 < contig_sizes[all_contigs2] - read_length)
    mask &= (all_positions1 >= 0) & (all_positions2 >= 0)
    logging.info(f"Removing {np.sum(mask == 0)} reads that are outside contigs")
    all_positions1 = all_positions1[mask]
    all_contigs1 = all_contigs1[mask]
    all_positions2 = all_positions2[mask]
    all_contigs2 = all_contigs2[mask]
    base_qualities = "I" * read_length
    return all_contigs1, all_contigs2, all_positions1, all_positions2, base_qualities


if __name__ == "__main__":
    typer.run(simulate_hic_from_file)
