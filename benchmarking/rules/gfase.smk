import os


rule run_gfase:
    input:
        unitig_graph=HifiasmResults.path() + "/hifiasm.hic.p_utg.gfa",
        sorted_hic_reads=HifiasmResults.path() + "/p_utg.sorted_by_read_name.bam"
    output:
        multiext(PhasingResults.path(phaser="gfase") + "/", "phase_0.fasta", "phase_1.fasta")
    params:
        out_dir=lambda wildcards, input, output: os.path.sep.join(output[0].split(os.path.sep)[:-1])
    shell:
        """
        rm -rf {params.out_dir} && 
        # remove old index file if has been created before
        rm -f {input.unitig_graph}i && 
        phase_contacts_with_monte_carlo -t 4 -i {input.sorted_hic_reads} -g {input.unitig_graph} -o {params.out_dir} --use_homology
        """