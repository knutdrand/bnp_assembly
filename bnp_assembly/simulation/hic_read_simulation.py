# Naive simulation of actualy HiC reads
import logging

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


def simulate(contigs_file_name: str, n_reads: int, read_length: int, fragment_size_mean: int, signal: float, out_base_name: str,
             read_name_prefix: str):
    contigs_file_name = bnp.open(contigs_file_name).read()

    # signal is ratio of reads that are not noise
    all_contigs1, all_contigs2, all_positions1, all_positions2, base_qualities = simulate_raw(contigs_file_name,
                                                                                              fragment_size_mean,
                                                                                              n_reads, read_length,
                                                                                              signal)

    for i, (contig_names, positions) in enumerate([[all_contigs1, all_positions1], [all_contigs2, all_positions2]]):
        out_sequences = []
        logging.info(f"Writing {len(contig_names)} reads contig number {i}")
        for read_number, (contig, position) in enumerate(zip(contig_names, positions)):
            sequence = contigs_file_name.sequence[contig][position:position + read_length]
            out_sequences.append((f"{read_name_prefix}{read_number}", sequence, base_qualities))

        data = bnp.datatypes.SequenceEntryWithQuality.from_entry_tuples(out_sequences)

        with bnp.open(out_base_name + f"{i + 1}.fq.gz", "w") as f:
            f.write(data)


def simulate_raw(contigs: bnp.datatypes.SequenceEntry, fragment_size_mean: float, n_reads: int, read_length: int, signal: float):
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
p

if __name__ == "__main__":
    typer.run(simulate)
