


rule test_hifiasm:
    input:
        HifiasmResults.from_flat_params(
            genome_build="sacCer3",
            individual="simulated",
            depth=1,
            dataset_size="small",
            n_reads=100
        ).file_path() + "/whatshap.tsv"
    output:
        touch("test_hifiasm")


rule test_yahs:
    input:
        ScaffoldingResults.from_flat_params(scaffolder="yahs").file_path() + "_scaffolds_final.fa"
    output:
        touch("test_yahs")


rule test_bnp_scaffolding:
    input:
        ScaffoldingResults.from_flat_params(scaffolder="bnp_scaffolding", depth=2, n_reads=50000).file_path() + "/scaffolds.fa"
    output:
        touch("test_bnp_scaffolding")


rule test_quast:
    input:
        [ScaffoldingResults.from_flat_params(dataset_size="medium", scaffolder=scaffolder, depth=5, n_reads=200000, extra_splits=8).file_path() + "_quast_report/report.tsv"
        for scaffolder in ["yahs", "bnp_scaffolding"]]
    output:
        touch("test_quast")


rule test_edison:
    input:
        [ScaffoldingResults.from_flat_params(dataset_size="small", scaffolder=scaffolder, depth=5, n_reads=50000, extra_splits=20).file_path() + ".edison.txt"
        for scaffolder in ["yahs", "bnp_scaffolding", "true_scaffolder"]]
    output:
        touch("test_edison")


rule evaluate:
    input:
        [
            ScaffoldingResults.from_flat_params(source="not_assembled", dataset_size="small", scaffolder=scaffolder, depth=10, n_reads=25000, extra_splits=20).file_path() + ".evaluation.txt"
            for scaffolder in ["yahs", "bnp_scaffolding", "true_scaffolder"]]
    output:
        touch("evaluate")


rule test_pbsim:
    input:
        SingleHaplotypeAndChromosomeHifiReads.from_flat_params().file_path() + "_0001.sam",
    output:
        touch("test_pbsim")



rule test_gfase:
    input:
        multiext(PhasingResults.from_flat_params(phaser="gfase", depth=2, n_reads=500000).file_path() + "/", "phase_0.fasta", "phase_1.fasta")
    output:
        touch("test_gfase")


rule small_test:
    input:
        ScaffolderAccuracy.from_flat_params(source="not_assembled", dataset_size="small", scaffolder='bnp_scaffolding', n_reads=40000, extra_splits=10, genome_build='simulated1').file_path()
    output:
        touch("small_test")

rule medium_test:
    input:
        ScaffolderAccuracy.from_flat_params(source="not_assembled", dataset_size="big", scaffolder='bnp_scaffolding', n_reads=100000, extra_splits=10, genome_build='sacCer3').file_path()
    output:
        touch("small_test")


rule test_accuracy:
    input:
        ScaffoldingResults.from_flat_params(
            genome_build="sacCer3",
            individual="simulated",
            dataset_size="small",
            depth="10",
            n_reads="40000",
            seed=123,
            source="not_assembled",
            extra_splits=20,
            split_on_n_ns=0,
            scaffolder="bnp_scaffolding",
        ).file_path() + "/accuracy.txt"
    output:
        touch("test_accuracy")
    run:
        with open(input[0]) as f:
            lines = f.readlines()
            assert float(lines[0].split()[1]) == 1.0
            assert float(lines[1].split()[1]) == 1.0


rule test_athalia_rosea_real_reads:
    input:
        ScaffoldingResults.from_flat_params(
            genome_build="athalia_rosea",
            individual="real",
            dataset_size="big",
            depth="10",
            n_reads="10000000",
            seed=1,
            source="not_assembled",
            extra_splits=50,
            split_on_n_ns=0,
            scaffolder="bnp_scaffolding",
        ).file_path() + "/accuracy.txt"
    output:
        touch("test_athalia_rosea_real_reads")
    run:
        with open(input[0]) as f:
            lines = f.readlines()
            # recall
            assert float(lines[0].split()[1]) >= 0.92
            # precision
            assert float(lines[1].split()[1]) >= 0.98
