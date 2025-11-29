import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from Bio import SeqIO
    import plotnine as p9
    from utils import data
    return data, p9, pl


@app.cell
def _(pl):
    taxons_df = pl.read_csv("dataset/Test/testsuperset-taxon-list.tsv", separator="\t")
    taxons_df
    return


@app.cell
def _(data):
    test_df = data.create_test_df()
    test_df
    return (test_df,)


@app.cell
def _(p9, pl, test_df):
    taxon_counts = test_df.group_by("taxon_id").len().sort("len", descending=True)

    _total_seqs = test_df.height
    _taxon_viz_df = taxon_counts.with_columns([
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
    return (taxon_counts,)


@app.cell
def _(taxon_counts):
    taxon_counts.head(90)
    return


@app.cell
def _(pl):
    phylum_df = pl.read_csv("dataset/taxon_phylum.tsv", has_header=False, separator="\t").rename({"column_1": "TaxonID", "column_2": "Phylum"})
    phylum_df
    return (phylum_df,)


@app.cell
def _(phylum_df, pl, taxon_counts, test_df):
    def create_hybrid_features(df, count_df, phylum_df, k=50):
        top_ids = count_df.head(k)["taxon_id"].to_list()
    
        joined_df = df.join(
            phylum_df, 
            left_on="taxon_id", 
            right_on="TaxonID", 
            how="left"
        )
    
        # Create the hybrid feature column
        # Logic: If in Top-K -> Use TaxonID (e.g., "9606")
        #        Else -> Use Phylum (e.g., "Chordata")
        #        If Phylum missing -> "Other"
        df_encoded = joined_df.with_columns(
            pl.when(pl.col("taxon_id").is_in(top_ids))
            .then(pl.col("taxon_id").cast(pl.String))
            .otherwise(
                pl.col("Phylum").fill_null("Other")
            )
            .alias("hybrid_taxon")
        )
    
        # One Hot Encode the hybrid column
        # This generates columns like:
        #   hybrid_taxon_9606 (Specific)
        #   hybrid_taxon_Chordata (General fallback)
        return df_encoded.to_dummies("hybrid_taxon")

    processed_df = create_hybrid_features(
        test_df,       
        taxon_counts,  
        phylum_df,     
        k=77           
    )

    processed_df.head(10)
    return


if __name__ == "__main__":
    app.run()
