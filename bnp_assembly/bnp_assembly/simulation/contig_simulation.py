import pickle
import bionumpy as bnp
import numpy as np
import logging
from ..agp import ScaffoldAlignments
from dataclasses import dataclass
import typing as tp
from .hic_read_simulation import MissingRegionsDistribution


@dataclass
class SimulatedContigs:
    contigs: bnp.datatypes.SequenceEntry
    alignment: ScaffoldAlignments
    inter_chromosome_splits: tp.Dict[str, tp.List[str]]
    intra_chromosome_splits: tp.Dict[str, tp.List[str]]


def random_spaced_locations(start, stop, n, min_space=1000):
    assert stop > min_space
    min_space = min(min_space, stop - start - 1)
    candidates = np.arange(start, stop, min_space)
    if len(candidates) < n:
        logging.warning(f"Did not manage to make {n} splits between {start} and "
                        f"{stop} with spacing {min_space}. Made {len(candidates)}")
    np.random.shuffle(candidates)
    return candidates[0:n]


def random_locations_with_some_short_intervals(start, stop, n_total, ratio_small=0, small_size=4000):
    assert start == 0
    n_small = int(n_total * ratio_small)
    n_big = n_total - n_small
    total_size = stop - start
    big_size = int((total_size - n_small * small_size) / n_big)
    sizes = np.array([big_size] * n_big + [small_size] * n_small)
    print(f"Making {n_big} big contigs of size {big_size} and {n_small} small contigs of size {small_size} between {start} and {stop}")
    np.random.shuffle(sizes)
    positions = np.cumsum(sizes)[:-1]  # remove last to get n intervals
    print("Positions: %s" % positions)
    return positions


class ContigSplitSimulator:
    def __init__(self, genome: bnp.datatypes.SequenceEntry, n_splits: int, min_contig_size: int = 15000, rng=np.random.default_rng()):
        self._genome = genome
        self._n_splits = n_splits
        self._min_contig_size = min_contig_size
        self._rng = rng

    def sample(self, shape=()) -> SimulatedContigs:
        return simulate_contigs_from_genome(self._genome, self._n_splits, self._min_contig_size, self._rng)


def simulate_contigs_from_genome(genome: bnp.datatypes.SequenceEntry, n_splits: int,
                                 min_contig_size: int = 6000, rng=np.random.default_rng(),
                                 also_split_at_ns=0,
                                 ratio_small_contigs: float = 0.0) -> SimulatedContigs:
    """
    If also_split_at_ns > 0, genome will be split at contiguous Ns of this number (which are meant to give original scaffolds)
    """
    if also_split_at_ns > 0:
        logging.info("Will split at contiguous Ns of length %d" % also_split_at_ns)

    new_contig_names = []
    new_contig_sequences = []
    new_contig_id = 0
    total_size = np.sum(genome.sequence.shape[1])
    weights = genome.sequence.shape[1] / total_size
    splits_at_contig = np.bincount(np.sort(
        rng.choice(np.arange(len(genome)), n_splits, p=weights)), minlength=len(genome))

    inter_chromosome_splits = []
    intra_chromosome_splits = []
    prev_contig_name = None

    scaffold_alignments = []

    for contig_id, n_random_splits in enumerate(splits_at_contig):
        old_contig_sequence = genome.sequence[contig_id]
        split_positions = np.array([], dtype=int)
        if also_split_at_ns > 0:
            is_n = ((old_contig_sequence == "N") | (old_contig_sequence == "n")).astype(int)
            split_positions = np.where(np.diff(is_n) == 1)[0] - 1
            logging.info("Found %d split positions based on N: %s" % (len(split_positions), split_positions))

        if n_random_splits == 0 and len(split_positions) == 0:
            # will not split anywhere
            contig_name = f"contig{new_contig_id}"
            new_contig_names.append(contig_name)
            new_contig_sequences.append(genome.sequence[contig_id])
            new_contig_id += 1
            end = len(old_contig_sequence)
            scaffold_alignments.append([genome.name[contig_id].to_string(), 0, end, contig_name, 0, end, "+"])
            continue

        random_split_positions = np.array([], dtype=int)
        if n_random_splits > 0:
            random_split_positions = np.sort(
                random_locations_with_some_short_intervals(0, len(old_contig_sequence), n_random_splits,
                                                           ratio_small=ratio_small_contigs,
                                                           small_size=min_contig_size)
            )

        logging.info("Introducing random split positions: %s" % random_split_positions)

        split_positions = np.sort(np.concatenate([split_positions, random_split_positions]))
        # add start and end of genome
        split_positions = np.insert(split_positions, 0, 0)
        split_positions = np.append(split_positions, len(old_contig_sequence))

        logging.info(f"Splitting contig {contig_id} between {split_positions}")

        for split_i, (start, end) in enumerate(zip(split_positions[0:-1], split_positions[1:])):
            #assert end-start >= min_contig_size, f"Contig {contig_id} split {split_i} between {start} and {end} is too small"
            logging.info(f"New contig at old contig {contig_id} between {start} and {end}")
            contig_name = f"contig{new_contig_id}"
            new_contig_names.append(contig_name)
            new_contig_sequences.append(genome.sequence[contig_id][start:end])
            new_contig_id += 1

            if split_i == 0:
                if prev_contig_name is not None:
                    inter_chromosome_splits.append((prev_contig_name, contig_name))
            else:
                intra_chromosome_splits.append((prev_contig_name, contig_name))

            prev_contig_name = contig_name
            scaffold_alignments.append([genome.name[contig_id].to_string(), start, end, contig_name, 0, end - start, "+"])

    new_fasta = bnp.datatypes.SequenceEntry.from_entry_tuples(
        zip(new_contig_names,
            trim_contig_sequences_for_ns(new_contig_sequences))
    )
    logging.info(f"Ended up with {len(new_fasta)} genome")
    logging.info(str(scaffold_alignments))
    return SimulatedContigs(new_fasta, ScaffoldAlignments.from_entry_tuples(scaffold_alignments),
                            inter_chromosome_splits, intra_chromosome_splits)


def introduce_unmappable_regions_to_contigs(contig_sequences: bnp.datatypes.SequenceEntry,
                                            prob_missing: float,
                                            missing_size: int,
                                            mapping_ratio=0):
    """
    Replaces contig_sequences inplace
    """

    contig_dict = {entry.name.to_string(): len(entry.sequence) for entry in contig_sequences}
    missing_regions = MissingRegionsDistribution(contig_dict, prob_missing, missing_size).sample()
    logging.info(missing_regions)

    for contig in contig_sequences:
        for interval in missing_regions:
            if interval.chromosome.to_string() == contig.name.to_string():
                idx = slice(int(interval.start), int(interval.stop))
                contig.sequence[idx] = make_unmappable_sequence(contig.sequence[idx], mapping_ratio)


def make_unmappable_sequence(base_sequence, mapping_ratio):
    if mapping_ratio == 0:
        return "N"
    else:
        is_n = np.random.choice([True, False], size=len(base_sequence), p=[1 - mapping_ratio, mapping_ratio])
        return np.where(is_n, bnp.as_encoded_array('N'*len(base_sequence)), base_sequence)


def trim_contig_sequences_for_ns(contig_sequences):
    # removes Ns at start and end
    return [
        seq[(seq != "N") & (seq != "n")] for seq in contig_sequences
    ]
