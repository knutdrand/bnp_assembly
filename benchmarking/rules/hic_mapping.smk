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
    threads: 40
    shell:
    # samtools view -f 1 is important to discard reads that are not properly paired
        """
    #samtools sort -n -O bam - \
    #arima_hic_mapping_pipeline/01_mapping_arima.sh {input.reads1} {input.reads2} {input.primary_assembly} {params.out_dir} {params.sra}
    bwa mem -t {config[n_threads]} -5SPM {input.primary_assembly} \
    {input.reads1} {input.reads2} \
    | samtools view -buS - | \
    sambamba sort --sort-by-name --show-progress -t {config[n_threads]} --memory-limit 16G -o - /dev/stdin  | \
    samtools fixmate -mr - -| \
    sambamba sort --show-progress -t {config[n_threads]} --memory-limit 16G -o - /dev/stdin  | \
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
        #"../envs/samtools.yml"
        "../envs/picard.yml"
    shell:
        """
        sambamba sort -t {config[n_threads]} --memory-limit 16G --sort-by-name --show-progress -o {output} {input}
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
        "{file}.sorted_by_read_name.bam.old"
    shell:
        """
        bnp_assembly sanitize-paired-end-bam {input} {output}
        """

# hic mapping without any sorting
rule map_hic2:
    input:
        reads1=HiCReads.path() + "/reads1.fq.gz",
        reads2=HiCReads.path() + "/reads2.fq.gz",
        primary_assembly=HifiasmResultsWithExtraSplits.path() + "/{assembly}.fa",
        bwa_index = multiext(HifiasmResultsWithExtraSplits.path() + "/{assembly}", ".fa.amb",".fa.ann",".fa.bwt",".fa.pac",".fa.sa")
    output:
        HifiasmResultsWithExtraSplits.path() + "/{assembly}.sorted_by_read_name.bam",
    conda:
        "../envs/hic_mapping.yml"
    shell:
        """
        bwa mem -t {config[n_threads]} -5SP {input.primary_assembly} {input.reads1} {input.reads2} | samblaster | samtools view - -@ 14 -S -h -b -F 3340 -o {output}
        """



rule chromap_index:
    input:
        #primary_assembly=HifiasmResultsWithExtraSplits.path() + "/{assembly}.fa",
        primary_assembly="{assembly}.fa",
    output:
        #HifiasmResultsWithExtraSplits.path() + "/{assembly}.chromap_index",
        "{assembly}.chromap_index",
    conda:
        "../envs/chromap.yml"
    shell:
        """
        chromap -i -r {input.primary_assembly} -o {output}
        """

rule map_hic_with_chromap:
    input:
        reads1=HiCReads.path() + "/reads1.fq.gz",
        reads2=HiCReads.path() + "/reads2.fq.gz",
        primary_assembly=HifiasmResultsWithExtraSplits.path() + "/{prefix}.fa",
        index=HifiasmResultsWithExtraSplits.path() + "/{prefix}.chromap_index",
    output:
        temp=temp(HifiasmResultsWithExtraSplits.path() + "/{prefix}.sorted_by_read_name.tmp.pairs"),
        pairs=HifiasmResultsWithExtraSplits.path() + "/{prefix}.sorted_by_read_name.pairs",
    conda:
        "../envs/chromap.yml"
    shell:
        """
        chromap --pairs -t {config[n_threads]} --preset hic -x {input.index} -r {input.primary_assembly} -1 {input.reads1} -2 {input.reads2} -o {output.temp} &&
        cat pairs.header {output.temp} > {output.pairs}
        """

rule map_hic_with_chromap_sam:
    input:
        reads1=HiCReads.path() + "/reads1.fq.gz",
        reads2=HiCReads.path() + "/reads2.fq.gz",
        primary_assembly=HifiasmResultsWithExtraSplits.path() + "/{prefix}.fa",
        index=HifiasmResultsWithExtraSplits.path() + "/{prefix}.chromap_index",
    output:
        pairs=HifiasmResultsWithExtraSplits.path() + "/{prefix}.sorted_by_read_name.sam",
    conda:
        "../envs/chromap.yml"
    shell:
        """
        chromap -t {config[n_threads]} --preset chip -x {input.index} -r {input.primary_assembly} -1 {input.reads1} -2 {input.reads2} -o {output}
        """



rule convert_pairs_to_short:
    input:
        pairs="{prefix}.pairs",
    output:
        short="{prefix}.short"
    shell:
        "grep -v '#' {input.pairs} | cut -f 2,3,4,5 > {output.short}"

