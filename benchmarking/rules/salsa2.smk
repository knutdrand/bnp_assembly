import os

rule bam_to_bed:
    input:
        "{file}.bam"
    output:
        "{file}.bed"
    conda:
        "../envs/bedtools.yml"
    shell:
        """
        bedtools bamtobed -i {input} > {output}
        """


rule run_salsa2:
    input:
        contigs=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa",
        contigs_index=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.fa.fai",
        hic_to_contig_mappings=HifiasmResultsWithExtraSplits.path() + "/hifiasm.hic.p_ctg.sorted_by_read_name.bed",
    output:
        scaffolds = ScaffoldingResults.path(scaffolder="salsa2") + "/scaffolds.fa",
        agp = ScaffoldingResults.path(scaffolder="salsa2") + "/scaffolds.agp",
    conda:
        "../envs/salsa2.yml"
    params:
        out_prefix = lambda wildcards,input,output: os.path.sep.join(output[0].split(os.path.sep)[0:-1])
    shell:
        """
        rm -rf {params.out_prefix} &&
        run_pipeline.py --clean no -p yes -a {input.contigs} -l {input.contigs_index} -b {input.hic_to_contig_mappings} -e CATG -o {params.out_prefix} && 
        mv {params.out_prefix}/scaffolds_FINAL.fasta {output.scaffolds} &&
        mv {params.out_prefix}/scaffolds_FINAL.agp {output.agp}
        """
