import logging
logging.basicConfig(level=logging.INFO)
import pickle
import bionumpy as bnp
import numpy as np
from bnp_assembly.simulation.contig_simulation import simulate_contigs_from_genome

rng = np.random.default_rng(1)
extra_splits = int(snakemake.wildcards.extra_splits)
contigs = bnp.open(snakemake.input[0]).read()

simulated = simulate_contigs_from_genome(contigs, extra_splits, rng=rng, also_split_at_ns=int(snakemake.wildcards.splits_on_n_ns))
new_fasta = simulated.contigs
agp = simulated.alignment

agp.to_agp(snakemake.output.agp)


with bnp.open(snakemake.output[0], "w") as f:
    f.write(new_fasta)


with open(snakemake.output[2], "wb") as f:
    pickle.dump({"inter_chromsome_splits": simulated.inter_chromosome_splits,
                "intra_chromosome_splits": simulated.intra_chromosome_splits}, f)
