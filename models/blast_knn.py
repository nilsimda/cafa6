import polars as pl
import sys
from pathlib import Path

def make_knn_preds_on_blast_results(
    results_path, 
    train_terms_path="dataset/Train/train_terms.tsv", 
    output_path=None,
    streaming=True,
    chunk_size=None
):
    """
    Generate KNN predictions from BLAST results with memory-efficient processing.
    
    Args:
        results_path: Path to BLAST format 6 output file
        train_terms_path: Path to training terms TSV file
        output_path: Optional path to save predictions (CSV format)
        streaming: Use streaming mode for ultra-low memory (default: True)
        chunk_size: Process in chunks of this many rows (for very large files)
    
    Returns:
        Polars DataFrame with predictions
    """
    # Define column names for BLAST format 6 output
    _blast_col_names = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore"
    ]
    
    print(f"Loading training terms from: {train_terms_path}")
    # Load training terms eagerly (usually much smaller than BLAST results)
    train_terms_df = pl.read_csv(train_terms_path, separator="\t")
    print(f"Loaded {train_terms_df.height} training terms")
    
    print(f"\nReading BLAST results from: {results_path}")
    print(f"Mode: {'Streaming' if streaming else 'Standard lazy'}")
    
    if streaming:
        # STREAMING MODE - Ultra-low memory for massive files
        # This processes data in batches without loading everything into memory
        return _process_streaming(
            results_path, 
            train_terms_df, 
            _blast_col_names, 
            output_path
        )
    
    elif chunk_size:
        # CHUNKED MODE - For files too large even for lazy mode
        return _process_chunked(
            results_path,
            train_terms_df,
            _blast_col_names,
            output_path,
            chunk_size
        )
    
    else:
        # LAZY MODE - Standard memory-efficient processing
        return _process_lazy(
            results_path,
            train_terms_df,
            _blast_col_names,
            output_path
        )


def _process_lazy(results_path, train_terms_df, blast_col_names, output_path):
    """Standard lazy execution - good for most cases."""
    
    # 1. Scan BLAST results lazily (doesn't load into memory yet)
    blast_lf = pl.scan_csv(
        results_path,
        separator="\t",
        has_header=False,
        new_columns=blast_col_names
    ).select(["qseqid", "sseqid", "bitscore"])  # Only keep needed columns
    
    print("Building lazy query plan...")
    
    # 2. Calculate total bitscore per query
    blast_denom_lf = blast_lf.group_by("qseqid").agg(
        pl.col("bitscore").sum().alias("total_bitscore")
    )
    
    # 3. Join with training terms and aggregate
    blast_knn_preds_lf = (
        blast_lf
        .join(
            train_terms_df.lazy().select(["EntryID", "term"]),
            left_on="sseqid",
            right_on="EntryID",
            how="inner"
        )
        .join(blast_denom_lf, on="qseqid", how="left")
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
    
    print("Executing query...")
    # Only now does the actual computation happen
    blast_knn_preds_df = blast_knn_preds_lf.collect()
    
    print(f"\nSuccess: Generated {blast_knn_preds_df.height} predictions.")
    print(f"Unique queries: {blast_knn_preds_df['qseqid'].n_unique()}")
    print(f"Unique terms: {blast_knn_preds_df['term'].n_unique()}")
    
    if output_path:
        print(f"\nSaving predictions to: {output_path}")
        blast_knn_preds_df.write_csv(output_path, include_header=False, separator="\t")
        print("Saved successfully!")
    
    return blast_knn_preds_df


def _process_streaming(results_path, train_terms_df, blast_col_names, output_path):
    """
    Streaming mode - processes data in batches.
    Best for extremely large files that don't fit in memory.
    """
    print("Using streaming mode for ultra-low memory usage...")
    
    # Use scan with streaming
    blast_lf = pl.scan_csv(
        results_path,
        separator="\t",
        has_header=False,
        new_columns=blast_col_names
    ).select(["qseqid", "sseqid", "bitscore"])
    
    # Calculate denominators
    blast_denom_lf = blast_lf.group_by("qseqid").agg(
        pl.col("bitscore").sum().alias("total_bitscore")
    )
    
    # Build the full query
    blast_knn_preds_lf = (
        blast_lf
        .join(
            train_terms_df.lazy().select(["EntryID", "term"]),
            left_on="sseqid",
            right_on="EntryID",
            how="inner"
        )
        .join(blast_denom_lf, on="qseqid", how="left")
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
    
    # If output path provided, sink directly to file without loading into memory
    if output_path:
        print(f"Streaming results directly to: {output_path}")
        blast_knn_preds_lf.sink_csv(output_path, include_header=False, separator="\t")
        print("Streaming complete!")
        # Return empty for streaming mode
        return pl.DataFrame(schema={"qseqid": pl.Utf8, "term": pl.Utf8, "knn_score": pl.Float64})
    else:
        print("Collecting results (streaming with batching)...")
        return blast_knn_preds_lf.collect(streaming=True)


def _process_chunked(results_path, train_terms_df, blast_col_names, output_path, chunk_size):
    """
    Manual chunked processing - for when even streaming isn't enough.
    Processes the file in explicit chunks.
    """
    print(f"Using chunked processing with chunk_size={chunk_size}")
    
    # This would require manual batching with polars.read_csv_batched
    # or reading the file in chunks manually
    raise NotImplementedError(
        "Chunked mode not yet implemented. Use streaming=True for most cases."
    )


def main():
    """Main entry point for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python script.py <blast_results_path> [train_terms_path] [output_path] [--no-streaming]")
        print("\nExample:")
        print("  python script.py blast_results.txt")
        print("  python script.py blast_results.txt dataset/Train/train_terms.tsv predictions.csv")
        print("  python script.py blast_results.txt dataset/Train/train_terms.tsv predictions.csv --no-streaming")
        sys.exit(1)
    
    results_path = sys.argv[1]
    train_terms_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "dataset/Train/train_terms.tsv"
    output_path = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
    streaming = '--no-streaming' not in sys.argv
    
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
        output_path=output_path,
        streaming=streaming
    )
    
    # Display sample of results (if not streaming to file)
    if predictions_df.height > 0:
        print("\n" + "="*60)
        print("Sample predictions (top 10 rows):")
        print("="*60)
        print(predictions_df.head(10))


if __name__ == "__main__":
    main()
