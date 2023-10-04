

rule run_bnp_scaffolding:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        hic_to_contig_mappings = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp_scaffolding") + '/logging/'
    output:
        fa = ScaffoldingResults.path(scaffolder="bnp_scaffolding") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="bnp_scaffolding") + "/scaffolds.agp"
    shell:
        "bnp_assembly scaffold {input.contigs} {input.hic_to_contig_mappings} {output.fa} --threshold 0.001 --bin-size 5000 --max-distance 100000 --logging-folder {params.log_folder} "


rule run_bnp_scaffolding_dynamic_heatmaps:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        hic_to_contig_mappings = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + '/logging/'
    output:
        fa = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/scaffolds.agp"
    shell:
        "bnp_assembly scaffold {input.contigs} {input.hic_to_contig_mappings} {output.fa} --threshold -100000000000000000000 "
        "--bin-size 5000  --logging-folder {params.log_folder} --distance-measure dynamic_heatmap "



rule run_bnp_scaffolding_nosplit:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        hic_to_contig_mappings = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp_scaffolding_nosplit") + '/logging/'
    output:
        fa = ScaffoldingResults.path(scaffolder="bnp_scaffolding_nosplit") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="bnp_scaffolding_nosplit") + "/scaffolds.agp"
    shell:
        """
        bnp_assembly scaffold {input.contigs} {input.hic_to_contig_mappings} {output.fa} --threshold -1000000000000000000000  --bin-size 5000 --max-distance 100000 --logging-folder {params.log_folder}
        """
