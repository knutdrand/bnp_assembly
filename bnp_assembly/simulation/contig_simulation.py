import pickle
import bionumpy as bnp
import numpy as np
import logging
from ..agp import ScaffoldAlignments
from dataclasses import dataclass

@dataclass
class SimulatedContigs:
    contigs: bnp.datatypes.SequenceEntry
    alignment: ScaffoldAlignments


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
                                 min_contig_size: int = 15000, rng=np.random.default_rng()) -> SimulatedContigs:
    
    new_contig_names = []
    new_contig_sequences = []
    new_contig_id = 0
    total_size = np.sum(genome.sequence.shape[1])
    weights = genome.sequence.shape[1]/total_size
    splits_at_contig = np.bincount(np.sort(
        rng.choice(np.arange(len(genome)), n_splits, p=weights)), minlength=len(genome))

    inter_chromsome_splits = []
    intra_chromosome_splits = []
    prev_contig_name = None

    scaffold_alignments = []

    for contig_id, n_splits in enumerate(splits_at_contig):
        if n_splits == 0:
            new_contig_names.append(f"contig{new_contig_id}")
            new_contig_sequences.append(genome.sequence[contig_id])
            new_contig_id += 1
            continue

        old_contig_sequence = genome.sequence[contig_id]
        split_positions = np.sort(random_spaced_locations(min_contig_size, len(old_contig_sequence)-min_contig_size,
                                                          n_splits, min_space=min_contig_size, rng=rng))
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
                    inter_chromsome_splits.append((prev_contig_name, contig_name))
            else:
                intra_chromosome_splits.append((prev_contig_name, contig_name))

            prev_contig_name = contig_name
            scaffold_alignments.append([genome.name[contig_id], start, end, contig_name, 0, end-start, "+"])


    new_fasta = bnp.datatypes.SequenceEntry.from_entry_tuples(
        zip(new_contig_names, new_contig_sequences)
    )
    logging.info(f"Ended up with {len(new_fasta)} genome")
    return SimulatedContigs(new_fasta, ScaffoldAlignments.from_entry_tuples(scaffold_alignments))