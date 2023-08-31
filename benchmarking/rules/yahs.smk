
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
    params:
        out_prefix = lambda wildcards, input, output: output[0].replace("_scaffolds_final.fa", "")
    shell:
        "yahs -q 10 -r 250,500,1000,5000,10000,20000,40000,50000 "
        "-o {params.out_prefix} "
        "{input.contigs} "
        "{input.hic_to_contig_mappings} && "
        "mv {params.out_prefix}_scaffolds_final.fa {output.scaffolds} && "
        "mv {params.out_prefix}_scaffolds_final.agp {output.agp} "


