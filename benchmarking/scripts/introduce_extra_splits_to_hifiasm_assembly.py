import logging
logging.basicConfig(level=logging.INFO)
import pickle
import bionumpy as bnp
import numpy as np
from bnp_assembly.simulation.contig_simulation import simulate_contigs_from_genome, \
    introduce_unmappable_regions_to_contigs

rng = np.random.default_rng(int(snakemake.wildcards.seed))
np.random.seed(int(snakemake.wildcards.seed))

extra_splits = int(snakemake.wildcards.extra_splits)
contigs = bnp.open(snakemake.input[0]).read()

simulated = simulate_contigs_from_genome(contigs, extra_splits, rng=rng, also_split_at_ns=int(snakemake.wildcards.split_on_n_ns),
                                         ratio_small_contigs=float(snakemake.wildcards.ratio_small_contigs),
                                         min_contig_size=int(snakemake.wildcards.min_contig_size))
new_fasta = simulated.contigs

print("Introducing unmappable regions", snakemake.wildcards.prob_low_mappability_region)
introduce_unmappable_regions_to_contigs(new_fasta, float(snakemake.wildcards.prob_low_mappability_region),
                                        int(snakemake.wildcards.mean_low_mappability_size),
                                        float(snakemake.wildcards.missing_region_mappability))

agp = simulated.alignment

agp.to_agp(snakemake.output.agp)


# shuffle contigs so that they are not in correct order, in case any method defaults to the order given
order = np.arange(0, len(new_fasta))
np.random.shuffle(order)
with bnp.open(snakemake.output[0], "w") as f:
    f.write(new_fasta[order])


with open(snakemake.output[2], "wb") as f:
    pickle.dump({"inter_chromsome_splits": simulated.inter_chromosome_splits,
                "intra_chromosome_splits": simulated.intra_chromosome_splits}, f)
