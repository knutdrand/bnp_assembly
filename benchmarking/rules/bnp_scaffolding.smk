

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
        """
        bnp_assembly scaffold {input.contigs} {input.hic_to_contig_mappings} {output.fa} --threshold 0.001 --logging-folder {params.log_folder} --bin-size 3567
        """

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
        bnp_assembly scaffold {input.contigs} {input.hic_to_contig_mappings} {output.fa} --threshold -1000 --logging-folder {params.log_folder} --bin-size 3567
        """
