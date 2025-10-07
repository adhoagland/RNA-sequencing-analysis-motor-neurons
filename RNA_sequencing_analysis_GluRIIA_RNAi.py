
"""
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
"""

# all comments below are intentionally lowercase, per user request

import os
import re
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ---- plotting prefs (lower verbosity, decent dpi) ----
sns.set_context("talk")
plt.rcParams["figure.dpi"] = 140

# ---- defaults for interactive runs (edit these once for spyder/jupyter) ----
DEFAULTS = {
    "fastq_dir": r"I:\M005951_MF",                  # folder containing .fastq.gz
    "hisat2_index_prefix": r"D:\ref\hisat2_dm6\dm6",# path prefix to hisat2 index (without .ht2 extension)
    "gtf": r"D:\ref\Drosophila_melanogaster.BDGP6.54.62.gtf.gz",  # gtf matching your genome build
    "outdir": r"D:\smartseq_out_hisat2",            # output directory
    "threads": 8,                                   # cpu threads for hisat2 and featurecounts
    "overwrite": False,                             # overwrite existing outputs
    "strandness": "none",                           # hisat2 --rna-strandness: 'none', 'fr', or 'rf' (smart-seq usually 'none')
}

# ---- r1/r2 name patterns to detect paired-end ----
R1_PATTERNS = ("_R1", "_1.", "_1_", ".R1.", ".r1.")
R2_PATTERNS = ("_R2", "_2.", "_2_", ".R2.", ".r2.")

# ---- arg parsing that works both in cli and interactive ide sessions ----
def get_args_interactive_aware():
    # build parser without required flags; fill from defaults when interactive
    ap = argparse.ArgumentParser(description="hisat2 + featurecounts + pydeseq2 pipeline (attp2 vs gluriia)")
    ap.add_argument("--fastq_dir", default=None)
    ap.add_argument("--hisat2_index_prefix", default=None)
    ap.add_argument("--gtf", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--threads", type=int, default=DEFAULTS["threads"])
    ap.add_argument("--overwrite", action="store_true", default=DEFAULTS["overwrite"])
    ap.add_argument("--strandness", default=DEFAULTS["strandness"], help="none, fr, or rf for hisat2 --rna-strandness")

    # detect interactive (spyder/jupyter) vs cli
    is_interactive = bool(getattr(sys, "ps1", None)) or ("SPYDER" in os.environ) or ("JPY_PARENT_PID" in os.environ)
    if is_interactive:
        ns = ap.parse_args([])  # ignore sys.argv
        for k in ("fastq_dir","hisat2_index_prefix","gtf","outdir"):
            if getattr(ns, k, None) in (None, ""):
                setattr(ns, k, DEFAULTS[k])
        return ns
    else:
        ns = ap.parse_args()
        missing = [k for k in ("fastq_dir","hisat2_index_prefix","gtf","outdir") if getattr(ns, k) in (None, "")]
        if missing:
            ap.error("missing required arguments: " + ", ".join("--"+m for m in missing))
        return ns

# ---- small helpers ----
def ensure_dir(p: Path):
    # make directory tree if needed
    p.mkdir(parents=True, exist_ok=True)

def is_gz_fastq(path: str) -> bool:
    # identify .fastq.gz or .fq.gz
    n = os.path.basename(path).lower()
    return n.endswith(".fastq.gz") or n.endswith(".fq.gz")

def guess_condition(name: str):
    # return 'attp2' or 'gluriia' by filename; else none
    n = name.lower()
    if "attp2" in n:
        return "Attp2"
    if "gluriia" in n:
        return "GluRIIA"
    return None

def sample_key_from_fname(path: str):
    # derive a sample key by removing r1/r2 tokens and extension
    base = os.path.basename(path)
    key = base
    key = re.sub(r"(_R1|_R2)\b", "", key, flags=re.IGNORECASE)
    key = re.sub(r"(_1\.|_2\.)", ".", key)
    key = re.sub(r"(_1_|_2_)", "_", key)
    key = re.sub(r"\.f(ast)?q\.gz$", "", key, flags=re.IGNORECASE)
    return key

def detect_pairing(files_for_sample):
    # detect paired or single-end from filenames
    r1, r2, se = [], [], []
    for f in files_for_sample:
        b = os.path.basename(f)
        if any(p in b for p in R1_PATTERNS):
            r1.append(f)
        elif any(p in b for p in R2_PATTERNS):
            r2.append(f)
        else:
            se.append(f)
    if r1 and r2:
        return "paired", sorted(r1), sorted(r2)
    return "single", sorted(se), []

def run(cmd, log_file=None):
    # run a subprocess and capture stdout/stderr to optional log file
    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}). see log: {log_file or 'stdout'}")
    return proc.stdout

# ---- main pipeline ----
def main():
    args = get_args_interactive_aware()

    fastq_dir = Path(args.fastq_dir)
    outdir = Path(args.outdir)
    aln_dir = outdir / "hisat2_alignments"
    plots_dir = outdir / "plots"
    ensure_dir(outdir); ensure_dir(aln_dir); ensure_dir(plots_dir)

    # 1) discover fastqs and group into samples
    fastqs = [str(p) for p in fastq_dir.rglob("*") if is_gz_fastq(str(p))]
    if not fastqs:
        print(f"no fastq(.gz) found under {fastq_dir}")
        sys.exit(1)

    samples = defaultdict(list)
    for fq in fastqs:
        samples[sample_key_from_fname(fq)].append(fq)

    rows = []
    for skey, files in samples.items():
        cond = guess_condition(skey)
        if cond is None:
            print(f"[warn] could not infer condition from name: {skey} (skipping)")
            continue
        pairing, r1s, r2s = detect_pairing(files)
        rows.append({
            "sample": skey,
            "condition": cond,
            "pairing": pairing,
            "n_files": len(files),
            "r1": json.dumps(r1s),
            "r2": json.dumps(r2s),
        })

    meta = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    if meta.empty:
        print("no samples with detectable conditions (attp2 / gluriia). check filenames.")
        sys.exit(1)

    meta.to_csv(outdir / "sample_sheet.csv", index=False)
    print(f"sample sheet written: {outdir/'sample_sheet.csv'}")
    print(meta[["sample","condition","pairing","n_files"]])

    # 2) run hisat2 to produce per-sample sam files
    for _, row in meta.iterrows():
        smp = row["sample"]
        pairing = row["pairing"]
        r1 = json.loads(row["r1"]); r2 = json.loads(row["r2"])
        sam_path = aln_dir / f"{smp}.sam"

        if sam_path.exists() and not args.overwrite:
            print(f"[skip] {sam_path} exists")
            continue
        if sam_path.exists() and args.overwrite:
            sam_path.unlink()

        cmd = [
            "hisat2",
            "-x", args.hisat2_index_prefix,
            "-p", str(args.threads),
            "-S", str(sam_path),
        ]

        # smart-seq is typically unstranded; if you know it's stranded, set --strandness to 'fr' or 'rf'
        if args.strandness.lower() in ("fr","rf"):
            cmd += ["--rna-strandness", args.strandness.upper()]

        if pairing == "paired":
            cmd += ["-1", ",".join(r1), "-2", ",".join(r2)]
        else:
            cmd += ["-U", ",".join(r1)]

        log_file = outdir / f"hisat2_{smp}.log"
        run(cmd, log_file=str(log_file))

    # 3) run featurecounts once across all sams to get a gene x sample count matrix
    sam_files = sorted(str(p) for p in aln_dir.glob("*.sam"))
    if not sam_files:
        print("no sam files produced; alignment must have failed.")
        sys.exit(1)

    counts_txt = outdir / "featureCounts_gene_counts.txt"
    if counts_txt.exists() and args.overwrite:
        counts_txt.unlink()

    fc_cmd = [
        "featureCounts",
        "-a", args.gtf,
        "-o", str(counts_txt),
        "-T", str(args.threads),
        "-t", "exon",      # count exons...
        "-g", "gene_id",   # ...group to gene_id
    ]

    # if any sample is paired, use -p and count read pairs
    if (meta["pairing"] == "paired").any():
        fc_cmd += ["-p", "--countReadPairs"]

    fc_cmd += sam_files
    run(fc_cmd, log_file=str(outdir / "featureCounts.log"))

    # 4) load featurecounts table -> counts dataframe
    with open(counts_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    from io import StringIO
    df_counts = pd.read_csv(StringIO("".join(lines)), sep="\t")
    df_counts = df_counts.rename(columns={"Geneid": "gene_id"})

    # featurecounts names columns by input filenames; map to sample names by stripping '.sam'
    col_map = {}
    for c in df_counts.columns:
        if c.endswith(".sam"):
            s = Path(c).stem
            col_map[c] = s
    df_counts = df_counts.rename(columns=col_map)

    sample_cols = [c for c in df_counts.columns if c in set(meta["sample"])]
    counts = df_counts.set_index("gene_id")[sample_cols].astype(int)
    counts.to_csv(outdir / "counts_gene_level.csv")

    # 5) prepare coldata and run pydeseq2
    coldata = meta[["sample","condition"]].set_index("sample").loc[counts.columns]

    dds = DeseqDataSet(
        counts=counts,
        clinical=coldata,
        design_factors="condition",
        refit_cooks=True
    )
    dds.deseq2()

    stats = DeseqStats(dds, n_cpus=max(1, min(args.threads, 8)))
    stats.summary()
    res = stats.results_df.copy()
    res.to_csv(outdir / "deseq2_results.csv")

    # 6) qc and results plots
    ensure_dir(plots_dir)

    # library sizes
    lib_sizes = counts.sum(axis=0).rename("library_size")
    ax = lib_sizes.sort_values().plot(kind="bar")
    ax.set_ylabel("total assigned reads (gene-level)")
    ax.set_title("library sizes")
    plt.tight_layout()
    plt.savefig(plots_dir / "01_library_sizes.png")
    plt.close()

    # library sizes by condition
    df_lib = pd.DataFrame({"library_size": lib_sizes, "condition": coldata["condition"]})
    plt.figure()
    sns.boxplot(data=df_lib, x="condition", y="library_size")
    sns.stripplot(data=df_lib, x="condition", y="library_size", dodge=True, alpha=0.6)
    plt.title("library sizes by condition")
    plt.tight_layout()
    plt.savefig(plots_dir / "02_library_sizes_by_condition.png")
    plt.close()

    # vst transform for pca and distances
    vst = dds.vst()
    vst_df = pd.DataFrame(vst, index=counts.index, columns=counts.columns)

    # pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(vst_df.T)
    pca_df = pd.DataFrame(pcs, columns=["pc1","pc2"], index=vst_df.columns)
    pca_df["condition"] = coldata["condition"]
    plt.figure()
    sns.scatterplot(data=pca_df, x="pc1", y="pc2", hue="condition", s=120)
    for s in pca_df.index:
        plt.text(pca_df.loc[s,"pc1"], pca_df.loc[s,"pc2"], s, fontsize=9, ha="left", va="bottom")
    ve = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"pc1 ({ve[0]:.1f}% var)"); plt.ylabel(f"pc2 ({ve[1]:.1f}% var)")
    plt.title("pca (vst)")
    plt.tight_layout()
    plt.savefig(plots_dir / "03_pca_vst.png")
    plt.close()

    # sample distance heatmap
    from scipy.spatial.distance import pdist, squareform
    dist_mat = squareform(pdist(vst_df.T, metric="euclidean"))
    dist_df = pd.DataFrame(dist_mat, index=vst_df.columns, columns=vst_df.columns)
    plt.figure(figsize=(6 + 0.2*len(dist_df), 5 + 0.2*len(dist_df)))
    sns.heatmap(dist_df, cmap="viridis", annot=False)
    plt.title("sample distances (vst, euclidean)")
    plt.tight_layout()
    plt.savefig(plots_dir / "04_sample_distance_heatmap.png")
    plt.close()

    # dispersion plot
    disp = pd.DataFrame({
        "baseMean": dds.varm["baseMean"],
        "dispGeneEst": dds.varm["dispGeneEst"],
        "dispFit": dds.varm["dispFit"],
        "dispersion": dds.varm["dispersion"]
    }, index=counts.index)
    plt.figure()
    plt.scatter(np.log10(disp["baseMean"] + 1e-8), np.log10(disp["dispersion"] + 1e-8), s=8, alpha=0.4)
    plt.xlabel("log10(baseMean)"); plt.ylabel("log10(dispersion)")
    plt.title("dispersion estimates")
    plt.tight_layout()
    plt.savefig(plots_dir / "05_dispersion.png")
    plt.close()

    # ma plot
    if {"baseMean","log2FoldChange","padj"}.issubset(res.columns):
        plt.figure()
        plt.scatter(np.log10(res["baseMean"] + 1e-8), res["log2FoldChange"], s=8, alpha=0.4)
        sig = (res["padj"] < 0.05) & (~res["padj"].isna())
        plt.scatter(np.log10(res.loc[sig,"baseMean"] + 1e-8), res.loc[sig,"log2FoldChange"], s=8, alpha=0.9)
        plt.axhline(0, ls="--", lw=1)
        plt.xlabel("log10(baseMean)"); plt.ylabel("log2fc (gluriia vs attp2)")
        plt.title("ma plot")
        plt.tight_layout()
        plt.savefig(plots_dir / "06_ma_plot.png")
        plt.close()

    # volcano
    if {"log2FoldChange","padj"}.issubset(res.columns):
        rv = res.assign(neglog10padj = -np.log10(res["padj"].clip(lower=1e-300)))
        plt.figure()
        plt.scatter(rv["log2FoldChange"], rv["neglog10padj"], s=8, alpha=0.4)
        sig = (rv["padj"] < 0.05) & (~rv["padj"].isna())
        plt.scatter(rv.loc[sig,"log2FoldChange"], rv.loc[sig,"neglog10padj"], s=8, alpha=0.9)
        plt.axvline(0, ls="--", lw=1)
        plt.xlabel("log2fc (gluriia vs attp2)"); plt.ylabel("-log10(padj)")
        plt.title("volcano")
        plt.tight_layout()
        plt.savefig(plots_dir / "07_volcano.png")
        plt.close()

    # violin plots for top de genes (vst)
    top = res.dropna(subset=["padj"]).sort_values("padj").head(12).reset_index().rename(columns={"index":"gene_id"})
    vst_long = (
        vst_df.loc[top["gene_id"]]
        .reset_index()
        .melt(id_vars="index", var_name="sample", value_name="vst")
        .rename(columns={"index":"gene_id"})
        .merge(coldata.reset_index(), on="sample", how="left")
    )
    plt.figure(figsize=(min(18, 2 + 1.2*len(top)), 6))
    sns.violinplot(data=vst_long, x="gene_id", y="vst", hue="condition", cut=0, inner="box")
    plt.xticks(rotation=60, ha="right")
    plt.title("top de genes (vst) – violin")
    plt.tight_layout()
    plt.savefig(plots_dir / "08_violin_top_de.png")
    plt.close()

    print(f"\ndone. outputs in: {outdir}")
    print(" - sample_sheet.csv")
    print(" - hisat2_alignments/*.sam")
    print(" - featurecounts_gene_counts.txt, counts_gene_level.csv")
    print(" - deseq2_results.csv")
    print(" - plots/*.png")

if __name__ == "__main__":
    main()
