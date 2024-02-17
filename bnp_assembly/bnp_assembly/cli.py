"""Console script for bnp_assembly."""
import os
from typing import Iterable

# todo
import numpy as np

import typer
import bionumpy as bnp

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.contig_graph import DirectedNode
from bnp_assembly.evaluation.compare_scaffold_alignments import ScaffoldComparison
from bnp_assembly.evaluation.debugging import ScaffoldingDebugger, analyse_missing_data
from bnp_assembly.evaluation.visualization import show_contigs_heatmap
from bnp_assembly.graph_objects import NodeSide
from bnp_assembly.heatmap import create_heatmap_figure
from bnp_assembly.input_data import FullInputData
from bnp_assembly.scaffolds import Scaffolds
from bnp_assembly.simulation.missing_data_distribution import MissingRegionsDistribution
from .io import get_genomic_read_pairs, PairedReadStream
from bnp_assembly.make_scaffold import make_scaffold, join as _join, split as _split
from bnp_assembly.max_distance import estimate_max_distance2
from .simulation import hic_read_simulation
import logging
from . import plotting

logging.basicConfig(level=logging.DEBUG)
app = typer.Typer()


def estimate_max_distance(contig_sizes: Iterable[int]):
    return int(np.median(list(contig_sizes)) / 4)


@app.command()
def scaffold(contig_file_name: str, read_filename: str, out_file_name: str, threshold: float = 0,
             logging_folder: str = None, bin_size: int = 5000, masked_regions: str = None, max_distance: int = None,
             distance_measure: str = "forbes3", n_bins_heatmap_scoring: int=10):
    logging.info(f"Using threshold {threshold}")

    if logging_folder is not None:
        register_logging(logging_folder)

    out_directory = os.path.sep.join(out_file_name.split(os.path.sep)[:-1])
    genome = bnp.Genome.from_file(contig_file_name)
    max_distance = set_max_distance(bin_size, genome, max_distance)

    logging.info("Getting genomic reads")
    read_stream = PairedReadStream.from_bam(genome, read_filename, mapq_threshold=20)
    input_data = FullInputData(genome, read_stream)
    logging.info("Making scaffold")
    scaffold = make_scaffold(input_data,
                             window_size=2500,
                             distance_measure=distance_measure,
                             threshold=threshold,
                             splitting_method='matrix',
                             bin_size=bin_size,
                             max_distance=max_distance,
                             n_bins_heatmap_scoring=n_bins_heatmap_scoring,
                             )
    alignments = scaffold.to_scaffold_alignments(genome, 1)
    alignments.to_agp(out_directory + "/scaffolds.agp")
    sequence_entries = scaffold.to_sequence_entries(genome.read_sequence())
    with bnp.open(out_file_name, "w") as f:
        f.write(sequence_entries)


@app.command()
def join(contig_file_name: str, read_filename: str, out_file_name: str):
    genome = bnp.Genome.from_file(contig_file_name)
    read_stream = PairedReadStream.from_bam(genome, read_filename, mapq_threshold=20)
    input_data = FullInputData(genome, read_stream)
    logging.info("Joining Path")
    scaffold = _join(input_data, 20)
    alignments = scaffold.to_scaffold_alignments(genome, 1)
    alignments.to_agp(out_file_name)


@app.command()
def split(contig_file_name: str, read_filename: str, joined_scaffold_filename: str, out_file_name: str):
    genome = bnp.Genome.from_file(contig_file_name)
    read_stream = PairedReadStream.from_bam(genome, read_filename, mapq_threshold=20)
    input_data = FullInputData(genome, read_stream)
    alignments = ScaffoldAlignments.from_agp(joined_scaffold_filename)
    scaffold = Scaffolds.from_scaffold_alignments(alignments)
    logging.info("Splitting Path")
    split_scaffolds = _split(input_data, scaffold)
    split_scaffolds.to_scaffold_alignments(genome, 1).to_agp(out_file_name)

def set_max_distance(bin_size, genome, max_distance):
    if max_distance is None:
        max_distance = estimate_max_distance2(genome.get_genome_context().chrom_sizes.values())
        max_distance = max(bin_size * 2, max_distance)
        max_distance -= max_distance % bin_size  # max distance must be divisible by bin size
        logging.info("Chose max distance from contig ends to be %d" % max_distance)
    return max_distance


def register_logging(logging_folder):
    plotting.register(splitting=plotting.ResultFolder(logging_folder + '/splitting'))
    plotting.register(joining=plotting.ResultFolder(logging_folder + '/joining'))
    plotting.register(dynamic_heatmaps=plotting.ResultFolder(logging_folder + '/dynamic_heatmaps'))
    plotting.register(missing_data=plotting.ResultFolder(logging_folder + '/missing_data'))


@app.command()
def heatmap(fasta_filename: str, interval_filename: str, agp_file: str, out_file_name: str, bin_size: int = 0):
    genome = bnp.Genome.from_file(fasta_filename, filter_function=None)
    locations_pair = get_genomic_read_pairs(genome, interval_filename)
    alignments = ScaffoldAlignments.from_agp(agp_file)
    fig, interaction_matrix = create_heatmap_figure(alignments, bin_size, genome, locations_pair)

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

    #plotting.register(missing_data=plotting.ResultFolder(out_path))

    truth_alignments = ScaffoldAlignments.from_agp(truth_agp)
    truth = Scaffolds.from_scaffold_alignments(truth_alignments)
    scaffolds = Scaffolds.from_scaffold_alignments(ScaffoldAlignments.from_agp(estimated_agp))

    genome = bnp.Genome.from_file(contigs_fasta)
    reads = PairedReadStream.from_bam(genome, mapped_reads_bam, mapq_threshold=20)

    debugger = ScaffoldingDebugger(scaffolds, truth, genome, reads, plotting_folder=out_path)
    # debugger.debug_edge(
    #    NodeSide("contig0", "+"),
    #    NodeSide("contig1", "-")
    # )
    debugger.debug_wrong_edges()
    debugger.finish()


@app.command()
def visualize_two_contigs(contigs_fasta: str, estimated_agp: str, truth_agp: str, mapped_reads_bam: str, out_path: str, contig_a: str, contig_b: str):

    truth_alignments = ScaffoldAlignments.from_agp(truth_agp)
    truth = Scaffolds.from_scaffold_alignments(truth_alignments)
    scaffolds = Scaffolds.from_scaffold_alignments(ScaffoldAlignments.from_agp(estimated_agp))

    genome = bnp.Genome.from_file(contigs_fasta)
    reads = PairedReadStream.from_bam(genome, mapped_reads_bam, mapq_threshold=20)

    debugger = ScaffoldingDebugger(scaffolds, truth, genome, reads, plotting_folder=out_path)
    debugger.make_heatmap_for_two_contigs(DirectedNode(contig_a, "+"), DirectedNode(contig_b, "+"), px=debugger.px)
    #debugger.finish()


@app.command()
def simulate_hic(contigs: str, n_reads: int, read_length: int, fragment_size_mean: int, signal: float,
                 out_base_name: str, read_name_prefix: str, mask_missing: bool = False, seed: int = 1):
    np.random.seed(seed)
    hic_read_simulation.simulate_hic_from_file(contigs, n_reads, read_length, fragment_size_mean, signal, out_base_name,
                                               read_name_prefix, do_mask_missing=mask_missing)


@app.command()
def evaluate_agp(estimated_agp_path: str, true_agp_path: str, out_file_name: str, contig_genome_fasta: str = None):
    contig_sizes = None
    if contig_genome_fasta is not None:
        # get contig sizes from fasta, and compute weighted scores also
        contig_sizes = bnp.Genome.from_file(contig_genome_fasta).get_genome_context().chrom_sizes

    estimated_agp = ScaffoldAlignments.from_agp(estimated_agp_path)
    true_agp = ScaffoldAlignments.from_agp(true_agp_path)
    comparison = ScaffoldComparison(estimated_agp, true_agp)
    comparison.set_contig_sizes(contig_sizes)

    missing_edges = comparison.missing_edges()
    comparison.describe_false_edges()
    comparison.describe_missing_edges()
    with open(out_file_name, "w") as f:
        f.write(f'edge_recall\t{comparison.edge_recall()}\n')
        f.write(f'edge_precision\t{comparison.edge_precision()}\n')
        if contig_sizes:
            f.write(f'weighted_edge_recall\t{comparison.weighted_edge_recall()}\n')
            f.write(f'weighted_edge_precision\t{comparison.weighted_edge_precision()}\n')

    with open(out_file_name + ".missing_edges", "w") as f:
        f.write('\n'.join([str(e) for e in missing_edges]))


@app.command()
def simulate_missing_regions(genome_file: str, out_file: str, prob_missing: float = 0.1, mean_size: int = 1000):
    genome = bnp.Genome.from_file(genome_file)
    missing_regions = MissingRegionsDistribution(genome.get_genome_context().chrom_sizes, prob_missing, mean_size)
    bnp.open(out_file, 'w').write(missing_regions.sample())


@app.command()
def sanitize_paired_end_bam(bam_file_name: str, out_file_name: str):
    """
    Checks that two and two lines are paired end reads (same name)
    Removes reads that are alone and not in pair
    """
    file = bnp.open(bam_file_name)
    # get all names as one single nparray
    all_names = []
    n = 0
    for chunk in file.read_chunks():
        names = np.array(chunk.name.tolist())
        all_names.append(names)
        n += len(names)
        logging.info(f"{n} reads processed")

    all_names = np.concatenate(all_names)

    keep = np.zeros(len(all_names), dtype=bool)
    # entries are fine if the name is equal to previous or next name
    keep[0:-1] = all_names[0:-1] == all_names[1:]
    keep[1:] |= (all_names[1:] == all_names[0:-1])
    logging.info(f"Keeping {np.sum(keep)} out of {len(keep)} reads")
    logging.info(f"Names not properly paired: {all_names[~keep]}")

    file = bnp.open(bam_file_name)
    out = bnp.open(out_file_name, "w")
    offset = 0
    for chunk in file.read_chunks():
        chunk_size = len(chunk)
        out.write(chunk[keep[offset:offset + chunk_size]])
        offset += chunk_size


def main():
    app()


if __name__ == "__main__":
    main()
