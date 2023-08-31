import os

rule run_hifiasm_with_hic_reads:
    input:
        hic1 = HiCReads.path() + "/reads1.fq.gz",
        hic2 = HiCReads.path() + "/reads2.fq.gz",
        hifi = HifiReads.path() + ".fq"

    output:
        multiext(HifiasmResults.path() + "/hifiasm", ".hic.hap1.p_ctg.gfa", ".hic.hap2.p_ctg.gfa", ".hic.p_ctg.gfa", ".hic.p_utg.gfa")
    params:
        out_base_name = lambda wildcards, input, output: os.path.sep.join(output[0].split(os.path.sep)[:-1]) + "/hifiasm"
    conda:
        "../envs/hifiasm.yml"
    shell:
        "hifiasm -o {params.out_base_name} -t6 --h1 {input.hic1} --h2 {input.hic2} {input.hifi}"




rule get_hifiasm_haplotypes_as_fasta:
    input:
        "{file}.p_{type}.gfa"
    output:
        "{file}.p_{type}.fa"
    conda:
        "../envs/gfatools.yml"
    shell:
        """
        gfatools gfa2fa {input} > {output}
        """



rule introduce_extra_splits_to_hifiasm_results:
    input:
        # can optinally just split the true haplotype sequence (or use assembled from hifi)
        lambda wildcards: HifiasmResults.path() + "/hifiasm.hic.p_ctg.fa" if wildcards.source == "assembled_from_hifi" else ReferenceGenome.path(file_ending="") + "/haplotype0.fa"
    output:
        fa=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        agp=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.agp",
        splits=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.edge_info"
    script:
        "../scripts/introduce_extra_splits_to_hifiasm_assembly.py"



rule dipcall:
    input:
        hap1 = HifiasmResults.path() + "/hifiasm.hic.hap1.p_ctg.fa",
        hap2 = HifiasmResults.path() + "/hifiasm.hic.hap2.p_ctg.fa",
        reference = ReferenceGenome.path(),
        reference_index = ReferenceGenome.path(file_ending = ".fa.fai"),
    output:
        HifiasmResults.path() + "/dipcall.mak",
        HifiasmResults.path() + "/dipcall.dip.vcf.gz",
    params:
        prefix = lambda wildcards, input, output: os.path.sep.join(output[0].split(os.path.sep)[:-1])
    shell:
        """
        dipcall.kit/run-dipcall -a {params.prefix}/dipcall {input.reference} {input.hap1} {input.hap2} > {output}  && 
        make --always-make -j2 -f {output}
        """


rule reaheader_dipcall_vcf:
    input:
        HifiasmResults.path() + "/dipcall.dip.vcf.gz"
    output:
        HifiasmResults.path() + "/dipcall.dip.cleaned.vcf.gz"
    conda:
        "../envs/bcftools.yml"
    shell:
        "echo '{wildcards.individual}\n' > {wildcards.individual}.txt && "
        "bcftools reheader --samples {wildcards.individual}.txt {input}  > {output}"


rule get_variants_for_dataset:
    input:
        variants=Individual.path() + "/variants.vcf.gz",
        index=Individual.path() + "/variants.vcf.gz.tbi"
    output:
        ReferenceGenome.path(file_ending="") + "/variants.vcf.gz"
    params:
        chromosomes = lambda wildcards: config["genomes"][wildcards.genome_build][wildcards.individual][wildcards.dataset_size]["chromosomes"]
    conda:
        "../envs/bcftools.yml"
    shell:
        """
        bcftools view --regions {params.chromosomes} -O z {input.variants} > {output}
        """

