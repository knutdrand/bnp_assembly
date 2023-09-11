import logging

import bionumpy as bnp

from bnp_assembly.agp import ScaffoldAlignments
from bnp_assembly.cli import app
from bnp_assembly.interface import SplitterInterface
from bnp_assembly.io import get_genomic_read_pairs
from bnp_assembly.scaffolds import Scaffolds


@app.command()
def generate_training(contig_file_name: str, read_filename: str, true_agp_path, out_file_name: str):
    true_agp = ScaffoldAlignments.from_agp(true_agp_path)
    genome = bnp.Genome.from_file(contig_file_name)
    logging.info("Getting genomic reads")
    reads = get_genomic_read_pairs(genome, read_filename, mapq_threshold=20)
    logging.info("Making scaffold")
    encoding = genome.get_genome_context().encoding
    contig_dict = genome.get_genome_context().chrom_sizes
    translation_dict = {int(encoding.encode(name).raw()): name for name in contig_dict}
    numeric_contig_dict = {int(encoding.encode(name).raw()): value for name, value in contig_dict.items()}
    numeric_locations_pair = reads.get_numeric_locations()
    path = next(Scaffolds.from_scaffold_alignments(true_agp)).to_contig_path(translation_dict)
    s = SplitterInterface(numeric_contig_dict, numeric_locations_pair, path,
                          max_distance=100000, bin_size=5000, threshold=0.2).split()
