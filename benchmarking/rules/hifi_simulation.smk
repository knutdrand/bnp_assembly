from itertools import product


def hifi_reads_files(wildcards):
    chromosomes = config["genomes"][wildcards.genome_build][wildcards.individual][wildcards.dataset_size]["chromosomes"].split(",")
    return [SingleHaplotypeAndChromosomeHifiReads.path(haplotype=haplotype, chromosome=c) + "_0001.ccs.bam"
            for haplotype, c in product([0, 1], chromosomes)]



rule chromosome_reference:
    input:
        "{path}/{ref}.fa"
    output:
        "{path}/{ref}/only_chromosome_{chromosome}.fa"
    conda:
        "../envs/samtools.yml"
    shell:
        "samtools faidx {input} {wildcards.chromosome} > {output}"



rule simulate_ccr_reads:
    input:
        haplotype_reference=Individual.path(file_ending="") + "/chromosome{chromosome}_haplotype{haplotype}_reference.fasta"
    output:
        SingleHaplotypeAndChromosomeHifiReads.path() + "_0001.sam",
    params:
        prefix = lambda wildcards, input, output: output[0].replace("_0001.sam", "")
    #conda:
    #    "../envs/pbsim.yml"
    shell:
        """
        pbsim \
        --strategy wgs \
        --method qshmm \
        --qshmm QSHMM-RSII.model \
        --difference-ratio 22:45:33 \
        --length-mean 15000 \
        --depth {wildcards.depth} \
        --genome {input.haplotype_reference} \
        --pass-num 10 \
        --prefix {params.prefix} 
        """ 


rule sam_to_bam:
    input:
        "{file}_0001.sam"
    output:
        "{file}_0001.bam"
    conda:
        "../envs/samtools.yml"
    shell:
        "samtools view {input} -o {output}"


rule make_hifi:
    input:
        SingleHaplotypeAndChromosomeHifiReads.path() + "_0001.bam",
    output:
        SingleHaplotypeAndChromosomeHifiReads.path() + "_0001.ccs.bam",
    conda:
        "../envs/ccs.yml"
    threads:
        10000  # hack to avoid running with other ccs commands, conflicting tmp files
    shell:
        "ccs -j {config[n_threads]} {input} {output}"


rule merge_hifi_bams:
    input:
        hifi_reads_files
    output:
        HifiReads.path() + ".bam"
    conda:
        "../envs/samtools.yml"
    shell:
        "samtools merge {output} {input}"
        
        
rule convert_hifi_bam_to_fq:
    input:
        HifiReads.path() + ".bam"
    output:
        HifiReads.path() + ".fq"
    conda:
        "../envs/bedtools.yml"
    shell:
        "bamToFastq -i {input} -fq {output}"



