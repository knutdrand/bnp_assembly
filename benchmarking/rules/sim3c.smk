
rule simulate_hic_for_haplotype:
    input:
        reference = ReferenceGenome.path(file_ending="") + "/haplotype{haplotype}.fa",
    output:
        reads1 = HiCReadsHaplotype.path() + "/1.fq.gz",
        reads2= HiCReadsHaplotype.path() + "/2.fq.gz",
    conda:
        "../envs/sim3c.yml"
    params:
        abundance_profile = lambda wildcards, input, output: "/".join(output.reads1.split("/")[:-1]) + "/profile.tsv",
        tmp_output = lambda wildcards, input, output: output.reads1.replace(".fq.gz", ".tmp")
    shell:
        """
        rm -f {params.abundance_profile} && 
        sim3C --prefix haplotype{wildcards.haplotype} \
        --create-cids \
        --seed 123 \
        --dist lognormal \
        -n {wildcards.n_reads} \
        -l 150 -e NlaIII \
        --insert-mean 10000 \
        --insert-sd 1000 \
        --insert-min 100 \
        -m hic {input.reference} {params.tmp_output} && 
        seqtk seq -1 {params.tmp_output} | gzip -c > {output.reads1} && 
        seqtk seq -2 {params.tmp_output} | gzip -c > {output.reads2} 
        """
