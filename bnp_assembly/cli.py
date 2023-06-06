"""Console script for bnp_assembly."""

import os

# todo
import numpy as np

import typer
import bionumpy as bnp

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from .io import get_read_pairs, get_genomic_read_pairs
from .path_finding import best_path, PathFinder
from .hic_distance_matrix import calculate_distance_matrices
from .scaffold import scaffold as scaffold_func
from .datatypes import GenomicLocationPair
from .interaction_matrix import InteractionMatrix
from .simulation import hic_read_simulation
import logging
from . import plotting

logging.basicConfig(level=logging.DEBUG)
app = typer.Typer()


@app.command()
def scaffold(contig_file_name: str, read_filename: str, out_file_name: str, threshold: float = 0,
             logging_folder: str = None, bin_size: int = 5000):
    '''
    Simple function

    >>> main()

    '''
    if logging_folder is not None:
        plotting.register(splitting=plotting.ResultFolder(logging_folder))
        plotting.register(joining=plotting.ResultFolder(logging_folder))
    out_directory = os.path.sep.join(out_file_name.split(os.path.sep)[:-1])
    genome = bnp.Genome.from_file(contig_file_name)
    encoding = genome.get_genome_context().encoding
    contig_dict = genome.get_genome_context().chrom_sizes
    translation_dict = {int(encoding.encode(name).raw()): name for name in contig_dict}
    numeric_contig_dict = {int(encoding.encode(name).raw()): value for name, value in contig_dict.items()}
    reads = get_read_pairs(genome, read_filename)

    paths = scaffold_func(numeric_contig_dict, reads, window_size=2500, distance_measure='forbes', threshold=threshold,
                          bin_size=bin_size)
    sequence_dict = genome.read_sequence()
    out_names = []
    out_sequences = []

    alignments = []

    for i, path in enumerate(paths):
        sequences = []
        scaffold_name = f'scaffold{i}_' + ':'.join(f'{dn.node_id}{dn.orientation}' for dn in path.directed_nodes)
        offset = 0
        print(path.directed_nodes)
        for j, dn in enumerate(path.directed_nodes):
            (contig_id, is_reverse) = dn.node_id, dn.orientation == '-'
            if j > 0:
                # adding 200 Ns between contigs
                sequences.append(bnp.as_encoded_array('N' * 200, bnp.encodings.ACGTnEncoding))
            seq = sequence_dict[translation_dict[contig_id]]
            if is_reverse:
                seq = bnp.sequence.get_reverse_complement(seq)
            sequences.append(bnp.change_encoding(seq, bnp.encodings.ACGTnEncoding))

            alignments.append(
                (scaffold_name, offset, offset + len(sequences[-1]),
                 translation_dict[contig_id], 0, len(sequences[-1]), "+" if not is_reverse else "-")
            )
            offset = sum((len(s) for s in sequences))
        print(scaffold_name)
        out_names.append(scaffold_name)
        out_sequences.append(np.concatenate(sequences))
    for path in paths:
        logging.info(path.directed_nodes)
    with bnp.open(out_file_name, "w") as f:
        f.write(bnp.datatypes.SequenceEntry.from_entry_tuples(
            zip(out_names, out_sequences)
        ))

    alignments = ScaffoldAlignments.from_entry_tuples(alignments)
    alignments.to_agp(out_directory + "/scaffolds.agp")


@app.command()
def heatmap(fasta_filename: str, interval_filename: str, agp_file: str, out_file_name: str, bin_size: int = 0):
    genome = bnp.Genome.from_file(fasta_filename, filter_function=None)
    print(bin_size)
    bin_size = max(bin_size, genome.size // 1000, 1000)
    print("Using bin size", bin_size)
    locations_pair = get_genomic_read_pairs(genome, interval_filename)
    interaction_matrix = InteractionMatrix.from_locations_pair(locations_pair, bin_size=bin_size)
    fig = interaction_matrix.plot()

    # add contig ids from agp file
    global_offset = genome.get_genome_context().global_offset
    alignments = ScaffoldAlignments.from_agp(agp_file)
    scaffold_offsets = global_offset.get_offset(alignments.scaffold_id)
    contig_offsets = (scaffold_offsets + alignments.scaffold_start) // bin_size
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=contig_offsets, ticktext=alignments.contig_id.tolist())),
    fig.update_xaxes(
        showgrid=True,
        ticks="outside",
        tickson="boundaries",
        ticklen=20
    )

    fig.show()
    fig.write_image(out_file_name)
    interaction_matrix.normalize_matrix().plot().show()


@app.command()
def simulate_hic(contigs: str, n_reads: int, read_length: int, fragment_size_mean: int, signal: float,
                 out_base_name: str, read_name_prefix: str):
    hic_read_simulation.simulate(contigs, n_reads, read_length, fragment_size_mean, signal, out_base_name,
                                 read_name_prefix)


@app.command()
def evaluate_agp(estimated_agp_path: str, true_agp_path: str):
    estimated_agp = ScaffoldAlignments.from_agp(estimated_agp_path)
    true_agp = ScaffoldAlignments.from_agp(true_agp_path)
    comparison = ScaffoldComparison(estimated_agp, true_agp)
    print(f'edge_recall\t{comparison.edge_recall()}')


def main():
    app()


if __name__ == "__main__":
    main()
