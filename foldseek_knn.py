# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from models.blast_knn import make_knn_preds_on_blast_results
    return make_knn_preds_on_blast_results, pl


@app.cell
def _(pl):
    blast_col_names = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", 
        "qstart", "qend", "sstart", "send", "evalue", "bitscore", "alignmtscore"
    ]

    foldseek_results_df = pl.read_csv("foldseek_work/result.m8", separator="\t", has_header=False, new_columns=blast_col_names)
    return blast_col_names, foldseek_results_df


@app.cell
def _(foldseek_results_df, pl):
    foldseek_processed_df = foldseek_results_df.with_columns(
        qseqid=pl.col("qseqid").str.split("-").list.get(1),
        sseqid=pl.col("sseqid").str.split("-").list.get(1)
    )
    return (foldseek_processed_df,)


@app.cell
def _(foldseek_processed_df):
    foldseek_processed_df
    return


@app.cell
def _(foldseek_processed_df):
    foldseek_avg_df = foldseek_processed_df.group_by(["qseqid", "sseqid"]).mean()
    foldseek_avg_df
    return (foldseek_avg_df,)


@app.cell
def _(foldseek_avg_df):
    foldseek_avg_df.write_csv("results/foldseek_results.m8", separator="\t", include_header=False)
    return


@app.cell
def _():
    return


@app.cell
def _(blast_col_names, pl):
    foldseek_loaded_df = pl.read_csv("results/foldseek_results.m8", separator="\t", has_header=False, new_columns=blast_col_names)

    foldseek_loaded_df
    return (foldseek_loaded_df,)


@app.cell
def _(foldseek_loaded_df, pl):
    foldseek_res_filtered_df = foldseek_loaded_df.filter(pl.col("evalue") < 1e-5)
    foldseek_res_filtered_df
    return (foldseek_res_filtered_df,)


@app.cell
def _(foldseek_res_filtered_df):
    foldseek_res_filtered_df.write_csv("results/foldseek_results_filtered.m8", separator="\t", include_header=False)
    return


@app.cell
def _(make_knn_preds_on_blast_results):
    make_knn_preds_on_blast_results(results_path="results/foldseek_results_filtered.m8", output_path="results/foldseek_knn_preds.tsv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
