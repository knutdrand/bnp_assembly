import pickle
import bionumpy as bnp
import numpy as np
import logging
from ..agp import ScaffoldAlignments
from dataclasses import dataclass
import typing as tp

@dataclass
class SimulatedContigs:
    contigs: bnp.datatypes.SequenceEntry
    alignment: ScaffoldAlignments
    inter_chromosome_splits: tp.Dict[str, tp.List[str]]
    intra_chromosome_splits: tp.Dict[str, tp.List[str]]


def random_spaced_locations(start, stop, n, min_space=1000, rng=np.random.default_rng()):
    assert stop > min_space
    min_space = min(min_space, stop - start - 1)
    candidates = np.arange(start, stop, min_space)
    if len(candidates) < n:
        logging.warning(f"Did not manage to make {n} splits between {start} and "
                        f"{stop} with spacing {min_space}. Made {len(candidates)}")
    np.random.shuffle(candidates)
    return candidates[0:n]


def simulate_contigs_from_genome(genome: bnp.datatypes.SequenceEntry, n_splits: int,
                                 min_contig_size: int = 15000, rng=np.random.default_rng(),
                                 also_split_at_ns=0) -> SimulatedContigs:
    """
    If also_split_at_ns > 0, genome will be split at contiguous Ns of this number (which are meant to give original scaffolds)
    """
    if also_split_at_ns > 0:
        logging.info("Will split at contiguous Ns of length %d" % also_split_at_ns)
    
    new_contig_names = []
    new_contig_sequences = []
    new_contig_id = 0
    total_size = np.sum(genome.sequence.shape[1])
    weights = genome.sequence.shape[1]/total_size
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
            new_contig_names.append(f"contig{new_contig_id}")
            new_contig_sequences.append(genome.sequence[contig_id])
            new_contig_id += 1
            continue


        random_split_positions = np.sort(random_spaced_locations(min_contig_size, len(old_contig_sequence)-min_contig_size,
                                                          n_random_splits, min_space=min_contig_size, rng=rng))
        logging.info("Introducing random split positions: %s" % random_split_positions)

        split_positions = np.sort(np.concatenate([split_positions, random_split_positions]))
        # add start and end of genome
        split_positions = np.insert(split_positions, 0, 0)
        split_positions = np.append(split_positions, len(old_contig_sequence))


        logging.info(f"Splitting contig {contig_id} between {split_positions}")

        for split_i, (start, end) in enumerate(zip(split_positions[0:-1], split_positions[1:])):
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
            scaffold_alignments.append([genome.name[contig_id], start, end, contig_name, 0, end-start, "+"])


    new_fasta = bnp.datatypes.SequenceEntry.from_entry_tuples(
        zip(new_contig_names,
            trim_contig_sequences_for_ns(new_contig_sequences))
    )
    logging.info(f"Ended up with {len(new_fasta)} genome")
    return SimulatedContigs(new_fasta, ScaffoldAlignments.from_entry_tuples(scaffold_alignments),
                            inter_chromosome_splits, intra_chromosome_splits)


def trim_contig_sequences_for_ns(contig_sequences):
    # removes Ns at start and end
    return [
        seq[(seq != "N") & (seq != "n")] for seq in contig_sequences
    ]
