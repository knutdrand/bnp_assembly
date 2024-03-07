import os

rule bwa_index:
    input:
        "{genome}.fa",
    output:
        idx=multiext("{genome}.fa", ".amb", ".ann", ".bwt", ".pac", ".sa"),
    params:
        algorithm="bwtsw",
    wrapper:
        "v1.21.2/bio/bwa/index"


rule map_hic:
    input:
        reads1=HiCReads.path() + "/reads1.fq.gz",
        reads2=HiCReads.path() + "/reads2.fq.gz",
        primary_assembly=HifiasmResultsWithExtraSplits.path() + "/{assembly}.fa",
        bwa_index = multiext(HifiasmResultsWithExtraSplits.path() + "/{assembly}", ".fa.amb",".fa.ann",".fa.bwt",".fa.pac",".fa.sa")
    output:
        HifiasmResultsWithExtraSplits.path() + "/{assembly}.bam"
    conda:
        #"../envs/hic_mapping.yml"
        "../envs/picard.yml"
    params:
        out_dir = lambda wildcards, input, output: os.path.sep.join(output[0].split(os.path.sep)[:-1]),  # replace(wildcards.assembly + ".bam", ""),
        sra = lambda wildcards, input, output: output[0].split(os.path.sep)[-1].replace(".bam", "")  # lambda wildcards: wildcards.assembly.replace(os.path.sep, "_")
    threads: 20
    shell:
    # samtools view -f 1 is important to discard reads that are not properly paired
        """
    #arima_hic_mapping_pipeline/01_mapping_arima.sh {input.reads1} {input.reads2} {input.primary_assembly} {params.out_dir} {params.sra}
	bwa mem -t {config[n_threads]} -5SPM {input.primary_assembly} \
	{input.reads1} {input.reads2} \
	| samtools view -buS - | samtools sort -n -O bam - \
	| samtools fixmate -mr - -| \
	samtools sort -O bam - | \
	samtools markdup -rsS - - | \
	samtools view -f 1 -O bam - > \
	{output}
    #"""


rule simulate_unmappable_regions:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/{assembly}.fa",
    output:
        regions = HifiasmResultsWithExtraSplits.path() + "/{assembly}.unmappable_regions.bed"
    shell:
        """
        bnp_assembly simulate-missing-regions {input.contigs} {output.regions} --prob-missing 0.3 --mean-size 1000
        """


rule remove_hic_reads_in_unmappable_regions:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/{assembly}.fa",
        regions = HifiasmResultsWithExtraSplits.path() + "/{assembly}.unmappable_regions.bed",
        reads = HifiasmResultsWithExtraSplits.path() + "/{assembly}.bam"
    output:
        filtered_reads = HifiasmResultsWithExtraSplits.path() + "/{assembly}.masked.bam"
    conda:
        "../envs/bedtools.yml"
    shell:
        """
        bedtools intersect -v -abam {input.reads} -b {input.regions} > {output.filtered_reads}
        """


rule sort_hic_mapped_reads_by_name:
    input:
        "{file}.bam"
    output:
        "{file}.sorted_by_read_name.unsanitized.bam"
    conda:
        "../envs/samtools.yml"
    shell:
        """
        samtools sort -n {input} -o {output}
        """


rule sort_hic_mapped_reads_by_position:
    input:
        "{file}.bam"
    output:
        "{file}.sorted_by_position.bam"
    conda:
        "../envs/samtools.yml"
    shell:
        """
        samtools sort {input} -o {output}
        """


rule sanitize_hic:
    input:
        "{file}.sorted_by_read_name.unsanitized.bam"
    output:
        "{file}.sorted_by_read_name.bam"
    shell:
        """
        bnp_assembly sanitize-paired-end-bam {input} {output}
        """

