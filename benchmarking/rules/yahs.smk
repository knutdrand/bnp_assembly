
rule run_yahs:
    input:
        contigs=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        hic_to_contig_mappings=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
    output:
        scaffolds = ScaffoldingResults.path(scaffolder="yahs") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="yahs") + "/scaffolds.agp",
    conda:
        "../envs/yahs.yml"
    benchmark:
        ScaffoldingResults.path(scaffolder="yahs") + "/benchmark.csv",
    params:
        out_prefix = lambda wildcards, input, output: output[0].replace("_scaffolds_final.fa", ""),
        args = lambda wildcards: config["yahs_parameters"][wildcards.genome_build]
    shell:
        "yahs -q 10 {params.args} "
        "-o {params.out_prefix} "
        "{input.contigs} "
        "{input.hic_to_contig_mappings} && "
        "mv {params.out_prefix}_scaffolds_final.fa {output.scaffolds} && "
        "mv {params.out_prefix}_scaffolds_final.agp {output.agp} "


