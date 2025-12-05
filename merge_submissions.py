#!/usr/bin/env python3
"""
Merge multiple CAFA submission files.

The primary submission is kept intact. Secondary submissions are merged in order,
adding only predictions for (protein, GO term) pairs NOT already present.

Usage:
    uv run merge_submissions.py \
        --primary submission_main.tsv \
        --secondary submission_backup1.tsv submission_backup2.tsv \
        --output merged_submission.tsv
"""

import argparse
import polars as pl


def load_submission(path: str) -> pl.DataFrame:
    """Load a CAFA submission TSV file."""
    return pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["EntryID", "term", "score"],
        schema={"EntryID": pl.Utf8, "term": pl.Utf8, "score": pl.Float64},
    )


def merge_submissions(
    primary_df: pl.DataFrame,
    secondary_dfs: list[pl.DataFrame],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Merge secondary submissions into primary.
    
    Only adds predictions for (EntryID, term) pairs not present in primary.
    Secondary submissions are processed in order, so earlier ones take precedence.
    
    Args:
        primary_df: Primary submission DataFrame
        secondary_dfs: List of secondary submission DataFrames (in priority order)
        verbose: Whether to print merge statistics
    
    Returns:
        Merged DataFrame
    """
    merged = primary_df.clone()
    
    if verbose:
        print(f"Primary: {len(merged):,} predictions")
    
    for i, secondary_df in enumerate(secondary_dfs):
        # Find pairs already in merged set
        existing_pairs = merged.select(["EntryID", "term"])
        
        # Anti-join to get only NEW pairs from secondary
        new_predictions = secondary_df.join(
            existing_pairs,
            on=["EntryID", "term"],
            how="anti"
        )
        
        if verbose:
            print(f"Secondary #{i+1}: {len(secondary_df):,} total, {len(new_predictions):,} new pairs added")
        
        # Add new predictions to merged set
        merged = pl.concat([merged, new_predictions])
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge CAFA submission files. Primary stays intact, "
                    "secondaries add only new (protein, term) pairs."
    )
    parser.add_argument(
        "--primary", "-p", required=True,
        help="Path to primary submission file (kept intact)"
    )
    parser.add_argument(
        "--secondary", "-s", nargs="+", required=True,
        help="Paths to secondary submission files (in priority order)"
    )
    parser.add_argument(
        "--output", "-o", default="merged_submission.tsv",
        help="Output path for merged submission (default: merged_submission.tsv)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Load primary submission
    if verbose:
        print(f"\nLoading primary: {args.primary}")
    primary_df = load_submission(args.primary)
    
    # Load secondary submissions
    secondary_dfs = []
    for path in args.secondary:
        if verbose:
            print(f"Loading secondary: {path}")
        secondary_dfs.append(load_submission(path))
    
    # Merge
    if verbose:
        print("\nMerging submissions...")
    merged_df = merge_submissions(primary_df, secondary_dfs, verbose=verbose)
    
    # Save
    if verbose:
        print(f"\nSaving merged submission to {args.output}")
    merged_df.write_csv(args.output, separator="\t", include_header=False)
    
    if verbose:
        print(f"\n✓ Merged submission: {len(merged_df):,} total predictions")
        added = len(merged_df) - len(primary_df)
        print(f"  - Primary: {len(primary_df):,}")
        print(f"  - Added from secondaries: {added:,}")


if __name__ == "__main__":
    main()

