import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from Bio import SeqIO
    import plotnine as p9
    return SeqIO, p9, pl


@app.cell
def _(pl):
    taxons_df = pl.read_csv("dataset/Test/testsuperset-taxon-list.tsv", separator="\t")
    taxons_df
    return


@app.cell
def _(SeqIO, pl):
    def create_test_df(fasta_path="dataset/Test/testsuperset.fasta"):

        data = [
            (*record.description.split(" ", 1), str(record.seq))
            for record in SeqIO.parse(fasta_path, "fasta")
        ]

        return pl.DataFrame(
            data,
            schema=[
                ("protein_id", pl.Utf8),
                ("taxon_id", pl.Int64),
                ("sequence", pl.Utf8),
            ],
            orient="row"
        )

    test_df = create_test_df()
    test_df
    return (test_df,)


@app.cell
def _(p9, pl, test_df):
    _taxon_counts = test_df.group_by("taxon_id").len().sort("len", descending=True)

    _total_seqs = test_df.height
    _taxon_viz_df = _taxon_counts.with_columns([
        (pl.col("len") / _total_seqs).alias("pct"),
        (pl.col("len").cum_sum() / _total_seqs).alias("cum_pct"),
        pl.arange(1, pl.len() + 1).alias("rank")
    ])

    print("Taxon ID Coverage Statistics:")
    print(f"Total unique taxa: {_taxon_viz_df.height}")
    print("-" * 40)
    for _threshold in [0.25, 0.50, 0.80, 0.90, 0.95, 0.99]:
        _n_taxa = _taxon_viz_df.filter(pl.col("cum_pct") >= _threshold)["rank"].min()
        print(f"Top {_n_taxa:4d} taxa cover {_threshold*100:.0f}% of sequences")
    print("-" * 40)

    (
        p9.ggplot(_taxon_viz_df, p9.aes(x="rank", y="cum_pct"))
        + p9.geom_line(color="#1f77b4", size=1.2)
        + p9.geom_hline(yintercept=[0.90, 0.95, 0.99], linetype="dashed", color="gray", alpha=0.7)
        + p9.annotate("text", x=1, y=0.91, label="90%", color="gray", ha="left", size=8)
        + p9.annotate("text", x=1, y=0.96, label="95%", color="gray", ha="left", size=8)
        + p9.scale_x_log10()
        + p9.labs(
            title="Cumulative Sequence Coverage by Taxon Rank",
            subtitle="Elbow plot to determine One-Hot Encoding cutoff size",
            x="Rank of Taxon (Log Scale, sorted by frequency)",
            y="Cumulative Fraction of Test Set Covered"
        )
        + p9.theme_minimal()
        + p9.theme(figure_size=(10, 6))
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
