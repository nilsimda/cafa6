# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.17.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    foldseek_preds_df = pl.read_csv("results/foldseek_preds.tsv", separator="\t", has_header=False)
    blast_preds_df = pl.read_csv("results/blast/blast_knn_maxseq10_1e-5_full.tsv", separator="\t", has_header=False)
    return blast_preds_df, foldseek_preds_df


@app.cell
def _(blast_preds_df, foldseek_preds_df, pl):
    ensemble_df = (
        foldseek_preds_df
        .rename({"column_1": "protein_id", "column_2": "go_term", "column_3": "foldseek_conf"})
        .join(
            blast_preds_df.rename({"column_1": "protein_id", "column_2": "go_term", "column_3": "blast_conf"}),
            on=["protein_id", "go_term"],
            how="full"
        )
        .with_columns([
            pl.coalesce(["foldseek_conf", pl.lit(0)]).alias("foldseek_conf"),
            pl.coalesce(["blast_conf", pl.lit(0)]).alias("blast_conf")
        ])
        .with_columns(
            pl.max_horizontal(["foldseek_conf", "blast_conf"]).alias("confidence")
        )
        .select(["protein_id", "go_term", "confidence"])
    )
    return (ensemble_df,)


@app.cell
def _(ensemble_df):
    ensemble_df.write_csv("results/folseek_blast_max_ensemble_preds.tsv", separator="\t", include_header=False)
    return


@app.cell
def _(blast_preds_df, foldseek_preds_df, pl):
    ensemble_avg_df = (
        foldseek_preds_df
        .rename({"column_1": "protein_id", "column_2": "go_term", "column_3": "foldseek_conf"})
        .join(
            blast_preds_df.rename({"column_1": "protein_id", "column_2": "go_term", "column_3": "blast_conf"}),
            on=["protein_id", "go_term"],
            how="full"
        )
        .with_columns([
            pl.coalesce(["foldseek_conf", pl.lit(0)]).alias("foldseek_conf"),
            pl.coalesce(["blast_conf", pl.lit(0)]).alias("blast_conf")
        ])
        .with_columns(
            ((pl.col("foldseek_conf") + pl.col("blast_conf")) / 2).alias("confidence")
        )
        .select(["protein_id", "go_term", "confidence"])
    )

    return (ensemble_avg_df,)


@app.cell
def _(ensemble_avg_df):
    ensemble_avg_df.write_csv("results/foldseek_blast_avg_ensemble_preds.tsv", separator="\t", include_header=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
