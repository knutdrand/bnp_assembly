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


rule make_interaction_matrix:
    input:
        contigs=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        hic_to_contig_mappings=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + '/logging/'
    output:
        matrix = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_{bin_size, \d+}.npz",
    shell:
        "bnp_assembly make-interaction-matrix {input.contigs} {input.hic_to_contig_mappings} {output.matrix} --bin-size {wildcards.bin_size} "


rule make_interaction_matrix_from_pairs:
    input:
        contigs=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        hic_to_contig_mappings=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.pairs",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + '/logging/'
    output:
        matrix = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_{bin_size, \d+}.pairs.npz",
    shell:
        "bnp_assembly make-interaction-matrix {input.contigs} {input.hic_to_contig_mappings} {output.matrix} --bin-size {wildcards.bin_size} "


rule plot_interaction_matrix:
    input:
        matrix = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_{bin_size}.npz",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + '/logging/'
    output:
        plot = touch(ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_{bin_size, \d+}.png")
    shell:
        "bnp_assembly plot-interaction-matrix {input.matrix}  "


rule plot_interaction_matrix_pairs:
    input:
        matrix = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_{bin_size}.pairs.npz",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + '/logging/'
    output:
        plot = touch(ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_{bin_size, \d+}.pairs.png")
    shell:
        "bnp_assembly plot-interaction-matrix {input.matrix}  "

rule get_cumulative_distribution:
    input:
        contigs=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        hic_to_contig_mappings=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + '/logging/'
    output:
        dist = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/cumulative_distribution.npz",
    shell:
        "bnp_assembly get-cumulative-distribution {input.contigs} {input.hic_to_contig_mappings} {output.dist}  "


rule run_bnp:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        hic_to_contig_mappings = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
        interaction_matrix = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_1000.npz",
        interaction_matrix_big = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_1000.npz",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp1k") + '/logging/'
    output:
        fa = ScaffoldingResults.path(scaffolder="bnp1k") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="bnp1k") + "/scaffolds.agp"
    shell:
        "bnp_assembly scaffold {input.contigs} {input.hic_to_contig_mappings} {output.fa} --threshold 20000000000000 "
        "--bin-size 5000  --logging-folder {params.log_folder} --distance-measure dynamic_heatmap --n-bins-heatmap-scoring 10 "
        "--interaction-matrix {input.interaction_matrix} --interaction-matrix-big {input.interaction_matrix_big}  "


rule run_bnp_iteration:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        interaction_matrix = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_1000.npz",
        agp = ScaffoldingResults.path(scaffolder="bnp1k") + "/scaffolds.agp"
    output:
        fa = ScaffoldingResults.path(scaffolder="bnp1k-iteration") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="bnp1k-iteration") + "/scaffolds.agp"
    shell:
        "bnp_assembly scaffold-iteration {input.agp} {input.contigs} {input.interaction_matrix} {output.fa} "



rule run_bnp10k:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        hic_to_contig_mappings = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
        interaction_matrix = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_10000.npz",
        interaction_matrix_big = ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_1000.npz",
    params:
        log_folder = ScaffoldingResults.path(scaffolder="bnp10k") + '/logging/'
    output:
        fa = ScaffoldingResults.path(scaffolder="bnp10k") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="bnp10k") + "/scaffolds.agp"
    shell:
        "bnp_assembly scaffold {input.contigs} {input.hic_to_contig_mappings} {output.fa} --threshold 20000000000000 "
        "--bin-size 5000  --logging-folder {params.log_folder} --distance-measure dynamic_heatmap --n-bins-heatmap-scoring 10 "
        "--interaction-matrix {input.interaction_matrix} --interaction-matrix-big {input.interaction_matrix_big}  "


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
        bnp_assembly scaffold {input.contigs} {input.hic_to_contig_mappings} {output.fa} --threshold 1000000000000000000000  \
        --bin-size 5000 --max-distance 100000 --logging-folder {params.log_folder} --distance-measure dynamic_heatmap
        """
