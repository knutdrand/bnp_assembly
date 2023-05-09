"""Console script for bnp_assembly."""
# todo
import numpy as np

import typer
import bionumpy as bnp
from .io import get_read_pairs, get_genomic_read_pairs
from .path_finding import best_path, PathFinder
from .hic_distance_matrix import calculate_distance_matrices
from .scaffold import scaffold as scaffold_func
from .datatypes import GenomicLocationPair
from .interaction_matrix import InteractionMatrix

app = typer.Typer()

@app.command()
def scaffold(contig_file_name: str, read_filename: str, out_file_name: str):
    '''
    Simple function

    >>> main()

    '''
    genome = bnp.Genome.from_file(contig_file_name)
    encoding = genome.get_genome_context().encoding
    contig_dict= genome.get_genome_context().chrom_sizes
    translation_dict = {int(encoding.encode(name).raw()): name for name in contig_dict}
    numeric_contig_dict = {int(encoding.encode(name).raw()): value for name, value  in contig_dict.items()}
    reads = get_read_pairs(genome, read_filename)
    paths = scaffold_func(numeric_contig_dict, reads, window_size=500)
    sequence_dict = genome.read_sequence()
    out_names = []
    out_sequences = []

    for i, path in enumerate(paths):
        sequences = []
        for contig_id, is_reverse in path.to_list():
            seq = sequence_dict[translation_dict[contig_id]]
            if is_reverse:
                seq = bnp.sequence.get_reverse_complement(seq)
            sequences.append(seq)
        # sequence_dict = {int(s.name.raw()): s.sequence for s in sequence_entires}
        out_names.append(f'contig{i}')
        out_sequences.append(np.concatenate(sequences))
        #print(np.concatenate(sequences).to_string())

    with bnp.open(out_file_name, "w") as f:
        f.write(bnp.datatypes.SequenceEntry.from_entry_tuples(
            zip(out_names, out_sequences)
        ))


@app.command()
def heatmap(fasta_filename: str, interval_filename: str, out_file_name: str, bin_size: int = 1000):
    genome = bnp.Genome.from_file(fasta_filename, filter_function=None)
    locations_pair = get_genomic_read_pairs(genome, interval_filename)
    interaction_matrix = InteractionMatrix.from_locations_pair(locations_pair, bin_size=bin_size)
    fig = interaction_matrix.plot()
    fig.show()
    fig.write_image(out_file_name)


def main():
    app()

if __name__ == "__main__":
    main()
