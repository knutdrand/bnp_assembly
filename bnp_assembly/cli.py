"""Console script for bnp_assembly."""
# todo
import numpy as np

import typer
import bionumpy as bnp
from .io import get_read_pairs
from .path_finding import best_path, PathFinder
from .hic_distance_matrix import calculate_distance_matrices
from .scaffold import scaffold
from .datatypes import GenomicLocationPair
from .interaction_matrix import InteractionMatrix

app = typer.Typer()

@app.command()
def scaffold(contig_file_name: str, read_filename: str):
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
    paths= scaffold(numeric_contig_dict, reads, window_size=500)
    sequence_dict = genome.read_sequence()
    for i, path in enumerate(paths):
        sequences = []
        for contig_id, is_reverse in path.to_list():
            seq = sequence_dict[translation_dict[contig_id]]
            if is_reverse:
                seq = bnp.sequence.get_reverse_complement(seq)
            sequences.append(seq)
        # sequence_dict = {int(s.name.raw()): s.sequence for s in sequence_entires}
        print(f'>contig{i}')
        print(np.concatenate(sequences).to_string())

@app.command()
def heatmap(fasta_filename: str, interval_filename: str):
    genome = bnp.Genome.from_file(fasta_filename)
    interval_pairs = get_read_pairs(genome, interval_filename)
    locations_pair = GenomicLocationPair(*(intervals.get_locations('center')
                                    for intervals in interval_pairs))
    interaction_matrix = InteractionMatrix.from_locations_pair(locations_pair)
    interaction_matrix.plot().show()

def main():
    app()
    # typer.run(scaffold_main)

if __name__ == "__main__":
    main()
