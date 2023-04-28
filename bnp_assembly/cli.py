"""Console script for bnp_assembly."""
# todo
import numpy as np

import typer
import bionumpy as bnp
from .io import get_read_pairs
from .path_finding import best_path, PathFinder
from .hic_distance_matrix import calculate_distance_matrices
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
    distance_matrix = calculate_distance_matrices(numeric_contig_dict, reads, window_size=500)

    path_finder = PathFinder(distance_matrix)
    #path = best_path(distance_matrix)
    paths = path_finder.run()

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

def main():
    typer.run(scaffold)

if __name__ == "__main__":
    main()
