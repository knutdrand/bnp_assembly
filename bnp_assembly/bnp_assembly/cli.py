"""Console script for bnp_assembly."""
import os
# todo
import numpy as np

import typer
import bionumpy as bnp

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from bnp_assembly.evaluation.debugging import ScaffoldingDebugger, analyse_missing_data
from bnp_assembly.graph_objects import NodeSide
from bnp_assembly.scaffolds import Scaffolds
from .io import get_genomic_read_pairs, PairedReadStream
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
    logging.info(f"Using threshold {threshold}")
    if logging_folder is not None:
        plotting.register(splitting=plotting.ResultFolder(logging_folder+'/splitting'))
        plotting.register(joining=plotting.ResultFolder(logging_folder+'/joining'))
    out_directory = os.path.sep.join(out_file_name.split(os.path.sep)[:-1])
    genome = bnp.Genome.from_file(contig_file_name)
    logging.info("Getting genomic reads")
    read_stream = PairedReadStream.from_bam(genome, read_filename, mapq_threshold=20)

    logging.info("Making scaffold")
    scaffold = make_scaffold(genome, read_stream, window_size=2500, distance_measure='forbes3', threshold=threshold, splitting_method='matrix', bin_size=2000)
    alignments = scaffold.to_scaffold_alignments(genome, 1)
    alignments.to_agp(out_directory + "/scaffolds.agp")
    sequence_entries = scaffold.to_sequence_entries(genome.read_sequence())
    with bnp.open(out_file_name, "w") as f:
        f.write(sequence_entries)



@app.command()
def heatmap(fasta_filename: str, interval_filename: str, agp_file: str, out_file_name: str, bin_size: int = 0):
    genome = bnp.Genome.from_file(fasta_filename, filter_function=None)
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
def missing_data(contig_fasta: str, reads: str, out_folder: str, bin_size: int = 1000):
    genome = bnp.Genome.from_file(contig_fasta, filter_function=None)
    reads = PairedReadStream.from_bam(genome, reads, mapq_threshold=20)
    analyse_missing_data(genome, reads, out_folder, bin_size)


@app.command()
def debug_scaffolding(contigs_fasta: str, estimated_agp: str, truth_agp: str, mapped_reads_bam: str, out_path: str):

    truth_alignments = ScaffoldAlignments.from_agp(truth_agp)
    truth = Scaffolds.from_scaffold_alignments(truth_alignments)
    scaffolds = Scaffolds.from_scaffold_alignments(ScaffoldAlignments.from_agp(estimated_agp))

    genome = bnp.Genome.from_file(contigs_fasta, filter_function=None)
    reads = PairedReadStream.from_bam(genome, mapped_reads_bam, mapq_threshold=20)

    debugger = ScaffoldingDebugger(scaffolds, truth, genome, reads, plotting_folder=out_path)
    #debugger.debug_edge(
    #    NodeSide("contig0", "+"),
    #    NodeSide("contig1", "-")
    #)
    debugger.debug_wrong_edges()
    debugger.finish()


@app.command()
def simulate_hic(contigs: str, n_reads: int, read_length: int, fragment_size_mean: int, signal: float,
                 out_base_name: str, read_name_prefix: str, seed: int = 1):
    np.random.seed(seed)
    hic_read_simulation.simulate(contigs, n_reads, read_length, fragment_size_mean, signal, out_base_name,
                                 read_name_prefix)


@app.command()
def evaluate_agp(estimated_agp_path: str, true_agp_path: str, out_file_name: str):
    estimated_agp = ScaffoldAlignments.from_agp(estimated_agp_path)
    true_agp = ScaffoldAlignments.from_agp(true_agp_path)
    comparison = ScaffoldComparison(estimated_agp, true_agp)
    missing_edges = comparison.missing_edges()
    with open(out_file_name, "w") as f:
        f.write(f'edge_recall\t{comparison.edge_recall()}\n')
        f.write(f'edge_precision\t{comparison.edge_precision()}\n')
    with open(out_file_name+".missing_edges", "w") as f:
        f.write('\n'.join([str(e) for e in missing_edges]))


def main():
    app()


if __name__ == "__main__":
    main()
