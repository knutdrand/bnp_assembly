


rule run_haphic:
    input:
        contigs=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        hic_to_contig_mappings=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.chromap.bam",
    output:
        scaffolds = ScaffoldingResults.path(scaffolder="haphic") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="haphic") + "/scaffolds.agp",
    conda:
        "../envs/haphic2.yml"
    benchmark:
        ScaffoldingResults.path(scaffolder="haphic") + "/benchmark.csv",
    params:
        n_chromosomes=lambda wildcards: config["genomes"][wildcards.genome_build]["n_chromosomes"],
    shell:
        """
        rm -rf 01.cluster/ &&
        rm -rf 02.reassign/ &&
        rm -rf 03.sort/ &&
        rm -rf 04.build/ &&
        external_tools/HapHiC/haphic pipeline {input.contigs} {input.hic_to_contig_mappings} {params.n_chromosomes} &&
        cp 04.build/scaffolds.fa {output.scaffolds} &&
        grep -v proximity_ligation 04.build/scaffolds.agp > {output.agp}
        """


