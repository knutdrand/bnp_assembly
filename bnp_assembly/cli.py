"""Console script for bnp_assembly."""
import dataclasses
import os
import typing as tp
# todo
import numpy as np

import typer
import bionumpy as bnp
from bionumpy.genomic_data import GenomicSequence

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.contig_graph import ContigPath
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from .io import get_read_pairs, get_genomic_read_pairs
from bnp_assembly.make_scaffold import make_scaffold_numeric as scaffold_func, make_scaffold
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
    reads = get_genomic_read_pairs(genome, read_filename)
    scaffold = make_scaffold(genome, reads, window_size=2500, distance_measure='forbes2', threshold=threshold)
    alignments = scaffold.to_scaffold_alignments(genome, 200)
    alignments.to_agp(out_directory + "/scaffolds.agp")
    sequence_entries = scaffold.to_sequence_entries(genome.read_sequence())
    with bnp.open(out_file_name, "w") as f:
        f.write(sequence_entries)
    return

    encoding = genome.get_genome_context().encoding
    contig_dict = genome.get_genome_context().chrom_sizes
    translation_dict = {int(encoding.encode(name).raw()): name for name in contig_dict}
    numeric_contig_dict = {int(encoding.encode(name).raw()): value for name, value in contig_dict.items()}
    reads = get_read_pairs(genome, read_filename)

    paths = scaffold_func(numeric_contig_dict, reads, window_size=2500, distance_measure='forbes', threshold=threshold,
                          bin_size=bin_size)
    sequence_dict = genome.read_sequence()
    paths_object = Paths(paths, translation_dict)
    alignments = paths_object.get_agp(contig_dict)
    alignments.to_agp(out_directory + "/scaffolds.agp")
    sequence_entries = paths_object.get_sequence_entries(sequence_dict)
    with bnp.open(out_file_name, "w") as f:
        f.write(sequence_entries)


@dataclasses.dataclass
class Paths:
    paths: tp.List[ContigPath]
    translation_dict: tp.Dict[int, str]
    padding: int = 200

    def get_agp(self, contig_dict: tp.Dict[str, int]):
        alignments = []
        for i, path in enumerate(self.paths):
            scaffold_name = self.get_scaffold_name(i, path)
            offset = 0
            for j, dn in enumerate(path.directed_nodes):
                (contig_id, is_reverse) = dn.node_id, dn.orientation == '-'
                contig_name = self.translation_dict[contig_id]
                length = contig_dict[contig_name]
                if j > 0:

                    length += self.padding
                alignments.append(
                    (scaffold_name, offset, offset + length,
                     contig_name, 0, length, "+" if not is_reverse else "-")
                )
                offset += length
        return ScaffoldAlignments.from_entry_tuples(alignments)

    def get_sequence_entries(self, sequence_dict: GenomicSequence):
        paths = self.paths
        out_names = []
        out_sequences = []

        for i, path in enumerate(paths):
            sequences = []
            scaffold_name = self.get_scaffold_name(i, path)
            offset = 0
            for j, dn in enumerate(path.directed_nodes):
                (contig_id, is_reverse) = dn.node_id, dn.orientation == '-'
                if j > 0:
                    # adding 200 Ns between contigs
                    sequences.append(bnp.as_encoded_array('N' * self.padding, bnp.encodings.ACGTnEncoding))
                seq = sequence_dict[self.translation_dict[contig_id]]
                if is_reverse:
                    seq = bnp.sequence.get_reverse_complement(seq)
                sequences.append(bnp.change_encoding(seq, bnp.encodings.ACGTnEncoding))

            out_names.append(scaffold_name)
            out_sequences.append(np.concatenate(sequences))
        return bnp.datatypes.SequenceEntry.from_entry_tuples(zip(out_names, out_sequences))

    def get_scaffold_name(self, i, path):
        return f'scaffold{i}_' + ':'.join(f'{dn.node_id}{dn.orientation}' for dn in path.directed_nodes)



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
def evaluate_agp(estimated_agp_path: str, true_agp_path: str, out_file_name: str):
    estimated_agp = ScaffoldAlignments.from_agp(estimated_agp_path)
    true_agp = ScaffoldAlignments.from_agp(true_agp_path)
    comparison = ScaffoldComparison(estimated_agp, true_agp)
    with open(out_file_name, "w") as f:
        f.write(f'edge_recall\t{comparison.edge_recall()}\n')


def main():
    app()


if __name__ == "__main__":
    main()
