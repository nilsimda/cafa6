import polars as pl
import sys
from pathlib import Path

def make_knn_preds_on_blast_results(results_path, train_terms_path="dataset/Train/train_terms.tsv", output_path=None):
    """
    Generate KNN predictions from BLAST results.
    
    Args:
        results_path: Path to BLAST format 6 output file
        train_terms_path: Path to training terms TSV file
        output_path: Optional path to save predictions (CSV format)
    
    Returns:
        Polars DataFrame with predictions
    """
    # Define column names for BLAST format 6 output
    _blast_col_names = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore"
    ]
    
    print(f"Loading training terms from: {train_terms_path}")
    train_terms_df = pl.read_csv(train_terms_path, separator="\t")
    print(f"Loaded {train_terms_df.height} training terms")
    
    print(f"\nReading BLAST results from: {results_path}")
    # 1. Load BLAST results
    _blast_lf = pl.read_csv(
        results_path,
        separator="\t",
        has_header=False,
        new_columns=_blast_col_names
    )
    print(f"Loaded {_blast_lf.height} BLAST hits")
    
    print("\nCalculating total bit score per query...")
    # 2. Calculate total bitscore per query (denominator for knn_score)
    _blast_denom_lf = _blast_lf.group_by("qseqid").agg(
        pl.col("bitscore").sum().alias("total_bitscore")
    )
    
    print("Joining with training terms and aggregating...")
    # 3. Join with Training Terms and Aggregate
    blast_knn_preds_df = (
        _blast_lf.join(
            train_terms_df.select(["EntryID", "term"]),
            left_on="sseqid",
            right_on="EntryID",
            how="inner"
        )
        .join(_blast_denom_lf, on="qseqid", how="left")
        .group_by(["qseqid", "term", "total_bitscore"])
        .agg([
            pl.col("bitscore").sum().alias("term_bitscore_sum")
        ])
        .with_columns(
            (pl.col("term_bitscore_sum") / pl.col("total_bitscore")).alias("knn_score")
        )
        .select(["qseqid", "term", "knn_score"])
        .sort(["qseqid", "knn_score"], descending=[False, True])
    )
    
    print(f"\nSuccess: Generated {blast_knn_preds_df.height} predictions.")
    print(f"Unique queries: {blast_knn_preds_df['qseqid'].n_unique()}")
    print(f"Unique terms: {blast_knn_preds_df['term'].n_unique()}")
    
    # Save if output path provided
    if output_path:
        print(f"\nSaving predictions to: {output_path}")
        blast_knn_preds_df.write_csv(output_path, include_header=False, separator="\t")
        print("Saved successfully!")
    
    return blast_knn_preds_df


def main():
    """Main entry point for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python script.py <blast_results_path> [train_terms_path] [output_path]")
        print("\nExample:")
        print("  python script.py blast_results.txt")
        print("  python script.py blast_results.txt dataset/Train/train_terms.tsv predictions.csv")
        sys.exit(1)
    
    results_path = sys.argv[1]
    train_terms_path = sys.argv[2] if len(sys.argv) > 2 else "dataset/Train/train_terms.tsv"
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Validate input files exist
    if not Path(results_path).exists():
        print(f"Error: BLAST results file not found: {results_path}")
        sys.exit(1)
    
    if not Path(train_terms_path).exists():
        print(f"Error: Training terms file not found: {train_terms_path}")
        sys.exit(1)
    
    # Run the function
    predictions_df = make_knn_preds_on_blast_results(
        results_path=results_path,
        train_terms_path=train_terms_path,
        output_path=output_path
    )
    
    # Display sample of results
    print("\n" + "="*60)
    print("Sample predictions (top 10 rows):")
    print("="*60)
    print(predictions_df.head(10))


if __name__ == "__main__":
    main()
