import os


rule naive_hic_simulation:
    input:
        reference=ReferenceGenome.path(file_ending="") + "/haplotype{haplotype}.fa",
    output:
        reads1 = HiCReadsHaplotype.path() + "/1.fq.gz",
        reads2=HiCReadsHaplotype.path() + "/2.fq.gz",
    params:
        output_base_name = lambda wildcards, input, output: os.path.sep.join(output.reads1.split(os.path.sep)[:-1]) + "/",
    shell:
        "bnp_assembly simulate-hic {input.reference} {wildcards.n_reads} 150 10000 0.6 {params.output_base_name} reads_haplotype{wildcards.haplotype}_ --seed {wildcards.seed} --mask-missing"


rule merge_hic_haplotype_reads:
    input:
        haplotype0 = HiCReadsHaplotype.path(individual="simulated", haplotype=0) + "/{pair}.fq.gz",
        haplotype1 = HiCReadsHaplotype.path(individual="simulated", haplotype=1) + "/{pair}.fq.gz",
    output:
        HiCReads.path(individual="simulated") + "/reads{pair}.fq.gz",
    shell:
        "zcat {input} | gzip -c > {output}"


rule download_real_hic_data:
    output:
        RawRealHicReads.path(individual="real") + "/reads{pair}.fq.gz",
    params:
        url = lambda wildcards: config["genomes"][wildcards.genome_build]["real"]["hic_data"][int(wildcards.pair)-1],
        n_lines = lambda wildcards: int(wildcards.n_reads) * 4 * 2,  # x 5 because subsample with random seed later
    shell:
        #"wget {params.url} -O - | gunzip | head -n {params.n_lines} | gzip -c > {output}"
        "curl -NL {params.url} 2>/dev/null | zcat | head -n {params.n_lines} | gzip -c > {output} || true"  # | gzip -c > {output}"



rule subsample_real_hic_data:
    input:
        RawRealHicReads.path(individual="real") + "/reads{pair}.fq.gz",
    output:
        HiCReads.path(individual="real") + "/reads{pair}.fq.gz",
    params:
        n_reads = lambda wildcards: int(wildcards.n_reads) // 2
    conda:
        "../envs/seqtk.yml",
    shell:
        "seqtk sample -s {wildcards.seed} {input} {params.n_reads} | gzip -c > {output}"

