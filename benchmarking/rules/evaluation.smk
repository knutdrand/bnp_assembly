import os
from collections import Counter, defaultdict


# A scaffolder giving correct results
rule truth_scaffolder:
    input:
        fa = ReferenceGenome.path(file_ending="") + "/haplotype0.fa",
        agp = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.agp",
    output:
        fa = ScaffoldingResults.path(scaffolder="true_scaffolder") + "/scaffolds.fa",
        agp= ScaffoldingResults.path(scaffolder="true_scaffolder") + "/scaffolds.agp"
    shell:
        """
        cp {input.fa} {output.fa}  && cp {input.agp} {output.agp}
        """


rule run_whatshap:
    input:
        truth = ReferenceGenome.path(file_ending="") + "/variants.vcf.gz",
        dipcall = HifiasmResults.path() + "/dipcall.dip.cleaned.vcf.gz"
    output:
        HifiasmResults.path() + "/whatshap.tsv"
    conda:
        "../envs/whatshap.yml"
    shell:
        """
        whatshap compare --names truth,sample --tsv-pairwise {output} {input} 
        """




rule make_heatmap:
    input:
        scaffolds = ScaffoldingResults.path() + "/scaffolds.fa",
        agp = ScaffoldingResults.path() + "/scaffolds.agp",
        hic_mapped_to_scaffolds = ScaffoldingResults.path() + "/scaffolds.sorted_by_read_name.bam",
        _ = ScaffoldingResults.path() + "/scaffolds.fa.fai",
    output:
        ScaffoldingResults.path() + "/heatmap.png"
    shell:
        """
        bnp_assembly heatmap {input.scaffolds} {input.hic_mapped_to_scaffolds} {input.agp} {output}
        """


rule make_heatmap2:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        agp = ScaffoldingResults.path() + "/scaffolds.agp",
        interaction_matrix= ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_2000.pairs.npz",
    output:
        ScaffoldingResults.path() + "/heatmap-{contig}-{window_size}.png"
    shell:
        """
        bnp_assembly heatmap2 {input.contigs} {input.agp} {input.interaction_matrix} {output} --contig {wildcards.contig} --window-size {wildcards.window_size}
        """


rule scaffold_heatmap:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        agp = ScaffoldingResults.path() + "/scaffolds.agp",
        interaction_matrix= ScaffoldingResults.path(scaffolder="bnp_scaffolding_dynamic_heatmaps") + "/interaction_matrix_2000.pairs.npz",
    output:
        ScaffoldingResults.path() + "/scaffold_heatmap.png"
    shell:
        """
        bnp_assembly scaffold-heatmap {input.contigs} {input.agp} {input.interaction_matrix} {output} 
        """

rule debug:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        scaffold_agp = ScaffoldingResults.path() + "/scaffolds.agp",
        true_agp = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.agp",
        hic_to_contig_mappings= HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
    output:
        scaffolds = ScaffoldingResults.path() + "/debug/report.html",
    params:
        output_path = ScaffoldingResults.path() + "/debug",
    shell:
        """
        bnp_assembly debug-scaffolding {input} {params.output_path} &&
        google-chrome {output.scaffolds}
        """


rule missing_data:
    input:
        contigs = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        hic_to_contig_mappings= HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
    output:
        report = ScaffoldingResults.path() + "/missing_data/report.html",
    params:
        output_path = ScaffoldingResults.path() + "/missing_data",
    shell:
        """
        bnp_assembly missing-data {input} {params.output_path} &&
        google-chrome {output.report}
        """



rule run_edison:
    input:
        assembly = ScaffoldingResults.path() + "/scaffolds.agp",
        true_reference = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.agp",
        #true_reference = ReferenceGenome.path(file_ending="") + "/haplotype0.fa"
    output:
        txt_report = ScaffoldingResults.path() + "/edison.txt",
        #alignment_viz = ScaffoldingResults.path() + "/alignment.pdf",
        #agp = ScaffoldingResults.path() + "/assembly.agp",
    conda: "../envs/edison.yml"
    threads:
        10000000  # hack: cannot be run in parallel because of temporary files
    params:
        #edison_agp_file = lambda wildcards, input, output: input.assembly.split(os.path.sep)[-1].replace(".fa", "_assembly.agp"),
        edison_pdf_alignment = lambda wildcards, input, output: input.assembly.split(os.path.sep)[-1].replace(".agp", "_alignment.pdf"),
    shell:
        """
        rm -f {output} &&
        python edison/edit_distance.py -a {input.assembly} -r {input.true_reference} > {output.txt_report} && 
        cat {output.txt_report}
        """
        #mv {params.edison_pdf_alignment} {output.alignment_viz} &&
        #&& gio open {params.edison_pdf_alignment}
        #mv {params.edison_agp_file} {output.agp} &&


# heatmap, edison
rule full_evaluation:
    input:
        ScaffoldingResults.path() + "/heatmap.png",
        ScaffoldingResults.path() + "/edison.txt",
        ScaffoldingResults.path() + "/quast_report/report.pdf",
    output:
        touch(ScaffoldingResults.path() + "/evaluation.txt")




rule accuracy:
    input:
        edison_results = ScaffoldingResults.path() + "/edison.txt"
    output:
        ScaffolderAccuracy.path()
    run:
        with open(input[0]) as f:
            line = [l for l in f if "Accuracy:" in l][0]
            accuracy = float(line.split(": ")[1].replace("%", ""))

        with open(output[0], "w") as f:
            f.write(str(accuracy) + "\n")



rule accuracy_bnp:
    input:
        scaffold_agp = ScaffoldingResults.path() + "/scaffolds.agp",
        true_agp = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.agp",
        contigs_fasta = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
    output:
        results = ScaffoldingResults.path() + "/accuracy.txt",
        missing_edges = ScaffoldingResults.path() + "/accuracy.txt.missing_edges",
    shell:
        """
        bnp_assembly evaluate-agp {input.scaffold_agp} {input.true_agp} {output.results} --contig-genome-fasta {input.contigs_fasta} \
        && cat {output.results}
        """


rule edge_recall:
    input:
        results = ScaffoldingResults.path() + "/accuracy.txt"
    output:
        ScaffolderEdgeRecall.path()
    run:
        with open(input[0]) as f:
            line = [l for l in f if "edge_recall" in l][0]
            print(line)
            accuracy = float(line.split()[1])

        with open(output[0],"w") as f:
            f.write(str(accuracy) + "\n")


rule edge_precision:
    input:
        results = ScaffoldingResults.path() + "/accuracy.txt"
    output:
        ScaffolderEdgePrecision.path()
    run:
        with open(input[0]) as f:
            line = [l for l in f if "edge_precision" in l][0]
            print(line)
            accuracy = float(line.split()[1])

        with open(output[0],"w") as f:
            f.write(str(accuracy) + "\n")


rule missing_edges:
    input:
        results=ScaffoldingResults.path() + "/accuracy.txt.missing_edges"
    output:
        ScaffolderMissingEdges.path()
    shell:
        "cp {input} {output}"


rule common_missing_edges:
    input:
        get_plot_input_files
    output:
        "plots/{plot_name}.missing_edges.txt"
    run:
        # Find common missing edges in all inputs
        counts = Counter()
        edge_to_file = defaultdict(list)

        for f in input:
            with open(f) as f:
                for line in f:
                    edge = line.strip()
                    counts[edge] += 1
                    edge_to_file[edge].append(f.name)

        print(counts)
        with open(output[0], "w") as f:
            for k, v in counts.items():
                f.write(f"{k}:{v}\n")
                print(f"{k}: {v}")
                if v >= 2:
                    print("\t" + "\n\t".join(edge_to_file[k]))

        print(edge_to_file)


rule make_pretextmap:
    input:
        bam=ScaffoldingResults.path() + "/scaffolds.chromap.sam"
    output:
        pretext=ScaffoldingResults.path() + "/scaffolds.pretext"
    conda:
        "../envs/pretextmap.yml"
    shell:
        """
        #samtools view -h {input.bam} | PretextMap -o {output} --sortby nosort --sortorder descend --mapq 10
        cat {input.bam} | PretextMap -o {output} --sortby nosort --sortorder descend --mapq 10
        """


rule pretextview:
    input:
        pretext=ScaffoldingResults.path() + "/scaffolds.pretext"
    output:
        touch(ScaffoldingResults.path() + "/scaffolds.pretextview")
    #conda:
    #	"../envs/pretextview.yml"
    shell:
        """
        ./PretextView {input.pretext}
        """


rule pretextview_via_chromap:
    input:
        reads1=HiCReads.path() + "/reads1.fq.gz",
        reads2=HiCReads.path() + "/reads2.fq.gz",
        primary_assembly=HifiasmResultsWithExtraSplits.path() + "/{prefix}.fa",
        index=HifiasmResultsWithExtraSplits.path() + "/{prefix}.chromap_index"
    output:
        pretext=HifiasmResultsWithExtraSplits.path() + "/{prefix}.direct.pretext",
    conda:
        "../envs/chromap.yml"
    shell:
        """
        chromap -t {config[n_threads]} -e 4 -q 1 --split-alignment -x {input.index} -r {input.primary_assembly} -1 {input.reads1} -2 {input.reads2} --SAM -o /dev/stdout |
        PretextMap -o {output} --sortby nosort --sortorder descend --mapq 10 --highres
        """


rule make_chrom_sizes:
    input:
        primary_assembly="{assembly}.fa.fai",
    output:
        chrom_sizes="{assembly}.chrom.sizes"
    shell:
        """
        cut -f 1,2 {input.primary_assembly} > {output.chrom_sizes}
        """

rule make_juicebox_hic:
    input:
        reads=ScaffoldingResults.path() + "/scaffolds.sorted_by_read_name.short",
        chrom_sizes=ScaffoldingResults.path() + "/scaffolds.chrom.sizes",
    output:
        hic=ScaffoldingResults.path() + "/scaffolds.hic"
    shell:
        """
        java -Xmx2g -jar hic_tools.3.30.00.jar pre {input.reads} {output.hic} {input.chrom_sizes}
        """


rule pretext_from_contigs_bam:
    input:
        contigs_bam = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bam",
        agp = ScaffoldingResults.path() + "/scaffolds.agp",
        contigs_fai = HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        scaffolds_fai=ScaffoldingResults.path() + "/scaffolds.fa.fai",
    output:
        scaffolds_fai=ScaffoldingResults.path() + "/scaffolds.from_contigs.pretext",
    conda:
        "../envs/pretextmap.yml"
    shell:
        """
        bnp_assembly translate-contig-bam-to-scaffold-coordinates {input}
        """

