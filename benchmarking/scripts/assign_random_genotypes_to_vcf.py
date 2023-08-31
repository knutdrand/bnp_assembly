# mason_variator outputs vcf without genotypes
# this script just assign a random genotype to each variant

import sys
import random

genotypes = ["0|0", "0|1", "1|0", "1|1"]


for line in sys.stdin:
    if line.startswith("#"):

        if line.startswith("#CHROM"):
            print('##FORMAT=<ID=GT,Number=1,Type=String,Description="Consensus Genotype across all datasets with called genotype">')

        print(line.strip())
        continue

    l = line.split()
    l[8] = "GT"
    l[9] = random.choice(genotypes)
    print("\t".join(l).strip())