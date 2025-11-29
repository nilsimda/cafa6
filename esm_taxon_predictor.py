import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from utils import data
    return data, np, pl


@app.cell
def _(data, pl):
    train_df = data.create_train_df()
    train_embs_df = pl.read_parquet("dataset/Train/esmc_600m_embs.parquet")
    return train_df, train_embs_df


@app.cell
def _(data, train_df):
    train_ohe_df = data.onehot_encode_taxon_ids(train_df)
    return (train_ohe_df,)


@app.cell
def _(train_embs_df, train_ohe_df):
    train_ohe_df.join(train_embs_df, on="protein_id", how="inner")
    return


@app.cell
def _(np, train_ohe_df):
    np.stack(train_ohe_df["taxon_ohe"].to_list())
    return


@app.cell
def _(train_df):
    from sklearn.model_selection import train_test_split

    X_train_df, X_val_df = train_test_split(train_df, random_state=3407)
    return (X_train_df,)


@app.cell
def _(X_train_df):
    X_train_df
    return


@app.cell
def _(pl):
    labels_df = pl.read_csv("dataset/Train/train_terms.tsv", separator="\t")
    labels_df
    return (labels_df,)


@app.cell
def _(labels_df, pl):
    labels_df.group_by("aspect").agg(
        pl.col("term").n_unique()
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
