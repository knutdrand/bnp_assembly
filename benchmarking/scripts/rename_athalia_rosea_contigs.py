import logging
logging.basicConfig(level=logging.INFO)
import sys

with open(sys.argv[1]) as f:
    for line in f:
        if not line.startswith(">"):
            print(line.strip())
            continue

        contig_name = line.split()[-1]
        print(f">{contig_name}")