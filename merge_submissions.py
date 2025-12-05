#!/usr/bin/env python3
"""
Merge multiple CAFA submission files.

Supports three merge modes:
- add_new: Keep primary intact, only add new (protein, term) pairs from secondaries
- average: Average scores for overlapping pairs, add new pairs from secondaries
- max: Take maximum score for overlapping pairs, add new pairs from secondaries

Usage:
    # Add new pairs only (default)
    uv run merge_submissions.py \
        --primary submission_main.tsv \
        --secondary submission_backup1.tsv submission_backup2.tsv \
        --output merged_submission.tsv

    # Average overlapping predictions
    uv run merge_submissions.py \
        --mode average \
        --primary submission_main.tsv \
        --secondary submission_backup1.tsv \
        --output averaged_submission.tsv

    # Take maximum score for overlaps
    uv run merge_submissions.py \
        --mode max \
        --primary submission_main.tsv \
        --secondary submission_backup1.tsv \
        --output max_submission.tsv
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


def merge_add_new(
    primary_df: pl.DataFrame,
    secondary_dfs: list[pl.DataFrame],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Merge by adding only new pairs from secondaries.
    
    Primary predictions are kept intact. Only (EntryID, term) pairs
    not present in primary are added from secondaries.
    Earlier secondaries take precedence over later ones.
    """
    merged = primary_df.clone()
    
    if verbose:
        print(f"Primary: {len(merged):,} predictions")
    
    for i, secondary_df in enumerate(secondary_dfs):
        existing_pairs = merged.select(["EntryID", "term"])
        
        new_predictions = secondary_df.join(
            existing_pairs,
            on=["EntryID", "term"],
            how="anti"
        )
        
        if verbose:
            print(f"Secondary #{i+1}: {len(secondary_df):,} total, {len(new_predictions):,} new pairs added")
        
        merged = pl.concat([merged, new_predictions])
    
    return merged


def merge_average(
    primary_df: pl.DataFrame,
    secondary_dfs: list[pl.DataFrame],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Merge by averaging scores for overlapping pairs.
    
    For (EntryID, term) pairs present in multiple submissions,
    the final score is the average of all scores.
    New pairs from secondaries are also included.
    """
    # Combine all submissions
    all_dfs = [primary_df] + secondary_dfs
    
    if verbose:
        print(f"Primary: {len(primary_df):,} predictions")
        for i, sec_df in enumerate(secondary_dfs):
            print(f"Secondary #{i+1}: {len(sec_df):,} predictions")
    
    combined = pl.concat(all_dfs)
    
    # Group by (EntryID, term) and average the scores
    merged = combined.group_by(["EntryID", "term"]).agg(
        pl.col("score").mean().alias("score")
    )
    
    if verbose:
        unique_pairs = len(merged)
        total_preds = sum(len(df) for df in all_dfs)
        overlaps = total_preds - unique_pairs
        print(f"Overlapping predictions averaged: {overlaps:,}")
    
    return merged


def merge_max(
    primary_df: pl.DataFrame,
    secondary_dfs: list[pl.DataFrame],
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Merge by taking maximum score for overlapping pairs.
    
    For (EntryID, term) pairs present in multiple submissions,
    the final score is the maximum of all scores.
    New pairs from secondaries are also included.
    """
    # Combine all submissions
    all_dfs = [primary_df] + secondary_dfs
    
    if verbose:
        print(f"Primary: {len(primary_df):,} predictions")
        for i, sec_df in enumerate(secondary_dfs):
            print(f"Secondary #{i+1}: {len(sec_df):,} predictions")
    
    combined = pl.concat(all_dfs)
    
    # Group by (EntryID, term) and take max score
    merged = combined.group_by(["EntryID", "term"]).agg(
        pl.col("score").max().alias("score")
    )
    
    if verbose:
        unique_pairs = len(merged)
        total_preds = sum(len(df) for df in all_dfs)
        overlaps = total_preds - unique_pairs
        print(f"Overlapping predictions (max taken): {overlaps:,}")
    
    return merged


MERGE_MODES = {
    "add_new": merge_add_new,
    "average": merge_average,
    "max": merge_max,
}


def main():
    parser = argparse.ArgumentParser(
        description="Merge CAFA submission files with different merge strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Merge modes:
  add_new  - Keep primary intact, only add new (protein, term) pairs (default)
  average  - Average scores for overlapping pairs
  max      - Take maximum score for overlapping pairs
"""
    )
    parser.add_argument(
        "--mode", "-m", choices=list(MERGE_MODES.keys()), default="add_new",
        help="Merge mode (default: add_new)"
    )
    parser.add_argument(
        "--primary", "-p", required=True,
        help="Path to primary submission file"
    )
    parser.add_argument(
        "--secondary", "-s", nargs="+", required=True,
        help="Paths to secondary submission files"
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
    
    # Merge using selected mode
    if verbose:
        print(f"\nMerging submissions (mode: {args.mode})...")
    
    merge_fn = MERGE_MODES[args.mode]
    merged_df = merge_fn(primary_df, secondary_dfs, verbose=verbose)
    
    # Save
    if verbose:
        print(f"\nSaving merged submission to {args.output}")
    merged_df.write_csv(args.output, separator="\t", include_header=False)
    
    if verbose:
        print(f"\n✓ Merged submission: {len(merged_df):,} total predictions")
        if args.mode == "add_new":
            added = len(merged_df) - len(primary_df)
            print(f"  - Primary: {len(primary_df):,}")
            print(f"  - Added from secondaries: {added:,}")
        else:
            print(f"  - Unique (protein, term) pairs from all inputs")


if __name__ == "__main__":
    main()
