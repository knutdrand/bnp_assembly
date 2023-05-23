"""Console script for bnp_assembly."""
import os

# todo
import numpy as np

import typer
import bionumpy as bnp

from bnp_assembly.agp import ScaffoldAlignments
from .io import get_read_pairs, get_genomic_read_pairs
from .path_finding import best_path, PathFinder
from .hic_distance_matrix import calculate_distance_matrices
from .scaffold import scaffold as scaffold_func
from .datatypes import GenomicLocationPair
from .interaction_matrix import InteractionMatrix
import logging
logging.basicConfig(level=logging.INFO)
app = typer.Typer()

@app.command()
def scaffold(contig_file_name: str, read_filename: str, out_file_name: str, threshold: float = 0):
    '''
    Simple function

    >>> main()

    '''
    out_directory = os.path.sep.join(out_file_name.split(os.path.sep)[:-1])
    genome = bnp.Genome.from_file(contig_file_name)
    encoding = genome.get_genome_context().encoding
    contig_dict= genome.get_genome_context().chrom_sizes
    translation_dict = {int(encoding.encode(name).raw()): name for name in contig_dict}
    numeric_contig_dict = {int(encoding.encode(name).raw()): value for name, value  in contig_dict.items()}
    reads = get_read_pairs(genome, read_filename)
    paths = scaffold_func(numeric_contig_dict, reads, window_size=2500, distance_measure='forbes', threshold=threshold)
    sequence_dict = genome.read_sequence()
    out_names = []
    out_sequences = []

    alignments = []

    for i, path in enumerate(paths):
        sequences = []
        scaffold_name = f'scaffold{i}_'+':'.join(f'{dn.node_id}{dn.orientation}' for dn in path.directed_nodes)
        offset = 0
        for j, (contig_id, is_reverse) in enumerate(path.to_list()):

            if j > 0:
                # adding 200 Ns between contigs
                sequences.append(bnp.as_encoded_array('N' * 200, bnp.encodings.ACGTnEncoding))
            seq = sequence_dict[translation_dict[contig_id]]
            if is_reverse:
                seq = bnp.sequence.get_reverse_complement(seq)
            sequences.append(bnp.change_encoding(seq, bnp.encodings.ACGTnEncoding))

            alignments.append(
                    (scaffold_name, offset+1, offset+len(sequences[-1]),
                     str(contig_id), 1, len(sequences[-1]), "+" if not is_reverse else "-")
            )
            offset = sum((len(s) for s in sequences))

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
    if bin_size == 0:
        bin_size = max(1000, genome.size // 1000)
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

def main():
    app()

if __name__ == "__main__":
    main()


