# RNA-sequencing-analysis-motor-neurons
Analysis of fastq RNA-sequencing results with feature counts, QC, differential expression analysis, and plots

windows-native smart-seq rna-seq pipeline using hisat2 + featurecounts + pydeseq2
- discovers fastq(.gz) in i:\m005951_mf
- aligns with hisat2 (single or paired)
- counts genes with featurecounts
- runs pydeseq2 for differential expression
- produces common qc and result plots (png) and csv outputs

usage (cli example):
  python smartseq_hisat2_featurecounts_pydeseq2.py ^
    --fastq_dir "i:\m005951_mf" ^
    --hisat2_index_prefix "d:\ref\hisat2_dm6\dm6" ^
    --gtf "d:\ref\drosophila_melanogaster.bdgp6.54.62.gtf.gz" ^
    --outdir "d:\smartseq_out_hisat2" ^
    --threads 8
