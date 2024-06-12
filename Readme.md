[![Tests]( https://github.com/knutdrand/bnp_assembly/actions/workflows/tests.yml/badge.svg)](https://github.com/ivargr/knutdrand/actions/workflows/tests.yml)
[![Tests]( https://github.com/knutdrand/bnp_assembly/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/ivargr/knutdrand/actions/workflows/benchmarks.yml)

This is a HiC-scaffolder that is under development. The scaffolder is based on similar ideas used in YAHS and Hap-HiC, but uses a different approach to score how contigs match, by estimating the distribution of HiC contacts between contig pairs of any size and comparing observed HiC contacts to that distribution.

The aim is to provide a scaffolder that does not require the user to do manual curation after scaffolding. 

This scaffolder should be fast and easy to use both for small and large genomes. All you need is contigs e.g. from Hifiasm and HiC data. 

## Installation

* Clone the repository
* Install the package with pip
```bash
cd bnp_assembly
pip install -e .
```

## Usage

```bash
bnp_assembly scaffold contigs.fa hic_reads.pairs out.fa
```

HiC reads should be sorted by read name and be in bam or pairs format (valid file endings are `.bam`, `.pairs`, and `.pa5`).

We recommend to use Chromap to map HiC-reads to the contigs as this is very fast (and does not yield worse scaffolding results than when mapping with BWA-MEM from our experience):

```bash
chromap -i -r contigs.fa -o index 
chromap --pairs --preset hic -x index -r contigs.fa -1 reads1.fq.gz -2 reads2.fq.gz -o hic_reads.pairs
``` 