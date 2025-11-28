import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    return (pl,)


@app.cell
def _(pl):
    blast_preds_df = pl.read_csv("results/blast/blast_knn_maxseq10_1e-5_full.tsv", separator="\t", has_header=False).rename({"column_1": "id", "column_2": "term", "column_3": "score"})
    esmc_preds_df = pl.read_csv("results/esmc/esmc600_knn_preds.tsv", separator="\t", has_header=False).rename({"column_1": "id", "column_2": "term", "column_3": "score"})
    esmc_preds_df
    return blast_preds_df, esmc_preds_df


@app.cell
def _(blast_preds_df, esmc_preds_df, pl):

    # Perform a full outer join to combine predictions from both models
    # Missing predictions are treated as a score of 0
    _joined_preds = blast_preds_df.join(
        esmc_preds_df,
        on=["id", "term"],
        how="full",
        suffix="_esmc"
    )

    # Define weight for the BLAST model (0.5 implies equal weighting)
    _blast_weight = 0.5

    ensemble_df = _joined_preds.select([
        pl.coalesce([pl.col("id"), pl.col("id_esmc")]).alias("id"),
        pl.coalesce([pl.col("term"), pl.col("term_esmc")]).alias("term"),
        (
            pl.col("score").fill_null(0.0) * _blast_weight +
            pl.col("score_esmc").fill_null(0.0) * (1 - _blast_weight)
        ).alias("score")
    ])

    ensemble_df
    return (ensemble_df,)


@app.cell
def _(ensemble_df):
    ensemble_df.write_csv("submission.tsv", include_header=False, separator="\t")
    return


@app.cell
def _(esmc_preds_df):
    esmc_preds_df["column_1", "column_2"].n_unique()
    return


app._unparsable_cell(
    r"""
    blast_preds_df.
    """,
    name="_"
)


@app.cell
def _(blast_preds_df):
    blast_preds_df["column_1", "column_2"].n_unique()
    return


@app.cell
def _(blast_preds_df, esmc_preds_df, pl):
    joined_preds = blast_preds_df.join(
        esmc_preds_df,
        on=["id", "term"],
        how="full",
        #suffix="_esmc"
    ).with_columns(mean=pl.mean_horizontal("score", "score_right"))


    joined_preds
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
