#!/usr/bin/env python3
"""
Generate CAFA submission file from trained Keras models.

Usage:
    uv run generate_submission.py \
        --model-c keras_models/best_mlp_model_c.keras \
        --model-f keras_models/best_mlp_model_f.keras \
        --model-p keras_models/best_mlp_model_p.keras \
        --output submission.tsv
"""

import argparse
import gc
import os

# Set backend before importing keras
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import polars as pl
from tqdm import tqdm

from cafaeval.parser import obo_parser
from utils import data


def get_label_order_dict(
    ontology_path: str = "dataset/Train/go-basic.obo",
    ia_path: str = "dataset/IA.tsv",
) -> dict[str, list[str]]:
    """Get the ordered list of GO terms for each aspect from the ontology."""
    ontologies = obo_parser(ontology_path, ("is_a", "part_of"), ia_path, True)
    
    return {
        "C": [term["id"] for term in ontologies["cellular_component"].terms_list],
        "F": [term["id"] for term in ontologies["molecular_function"].terms_list],
        "P": [term["id"] for term in ontologies["biological_process"].terms_list],
    }


def load_test_features(
    embeddings_path: str = "dataset/Test/test_esmc_600m.parquet",
    embedding_col: str = "esmc_600m_embedding",
    taxon_topk: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test set features (embeddings + taxon one-hot encoding).
    
    Args:
        embeddings_path: Path to test embeddings parquet file
        embedding_col: Column name for embeddings in the parquet file
        taxon_topk: Number of top taxon IDs for one-hot encoding
    
    Returns:
        Tuple of (X_test features array, protein_ids array)
    """
    print("Loading test data...")
    test_df = data.create_test_df()
    test_embs_df = pl.read_parquet(embeddings_path)
    
    print(f"  Test samples: {len(test_df)}")
    print(f"  Test embeddings: {len(test_embs_df)}")
    
    # One-hot encode taxon IDs
    print(f"  Encoding taxon IDs (top-k={taxon_topk})...")
    test_ohe_df = data.onehot_encode_taxon_ids(test_df, k=taxon_topk)
    
    # Join with embeddings
    test_ohe_embs_df = test_ohe_df.join(test_embs_df, on="protein_id", how="inner")
    print(f"  Samples with embeddings: {len(test_ohe_embs_df)}")
    
    # Create feature matrix
    test_ohes = np.stack(test_ohe_embs_df["taxon_ohe"].to_list())
    test_embs = np.stack(test_ohe_embs_df[embedding_col].to_list())
    X_test = np.hstack([test_ohes, test_embs])
    
    protein_ids = test_ohe_embs_df["protein_id"].to_numpy()
    
    print(f"  Feature matrix shape: {X_test.shape}")
    return X_test, protein_ids


def generate_predictions_for_aspect(
    aspect: str,
    model_path: str,
    X_input: np.ndarray,
    protein_ids: np.ndarray,
    label_list: list[str],
    threshold: float = 0.01,
    chunk_size: int = 5000,
    batch_size: int = 128,
) -> pl.DataFrame:
    """
    Generate predictions for a specific GO aspect using a saved Keras model.
    
    Processes in chunks to optimize memory usage.
    
    Args:
        aspect: GO aspect ("C", "F", or "P")
        model_path: Path to saved Keras model
        X_input: Feature matrix for all test samples
        protein_ids: Array of protein IDs
        label_list: Ordered list of GO term IDs for this aspect
        threshold: Minimum score threshold to include in predictions
        chunk_size: Number of samples to process per chunk (for memory)
        batch_size: Batch size for model prediction
    
    Returns:
        DataFrame with columns: EntryID, term, score
    """
    print(f"\n[{aspect}] Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Convert label list to array for fast indexing
    term_arr = np.array(label_list)
    
    all_ids = []
    all_terms = []
    all_scores = []
    
    num_samples = len(protein_ids)
    print(f"[{aspect}] Generating predictions for {num_samples} samples...")
    
    # Iterate in chunks to avoid OOM
    for start_idx in tqdm(range(0, num_samples, chunk_size), desc=f"[{aspect}]"):
        end_idx = min(start_idx + chunk_size, num_samples)
        
        # Prepare batch
        X_chunk = X_input[start_idx:end_idx]
        ids_chunk = protein_ids[start_idx:end_idx]
        
        # Predict
        y_pred = model.predict(X_chunk, verbose=0, batch_size=batch_size)
        
        # Filter predictions above threshold
        rows, cols = np.where(y_pred > threshold)
        scores = y_pred[rows, cols]
        
        # Append valid predictions
        if len(scores) > 0:
            all_ids.append(ids_chunk[rows])
            all_terms.append(term_arr[cols])
            all_scores.append(scores)
        
        # Clean up chunk memory
        del y_pred, rows, cols, scores, X_chunk
        gc.collect()
    
    # Explicit memory cleanup
    del model
    gc.collect()
    
    print(f"[{aspect}] Concatenating results...")
    if not all_ids:
        df = pl.DataFrame({
            "EntryID": [],
            "term": [],
            "score": []
        }, schema={"EntryID": pl.Utf8, "term": pl.Utf8, "score": pl.Float64})
    else:
        df = pl.DataFrame({
            "EntryID": np.concatenate(all_ids),
            "term": np.concatenate(all_terms),
            "score": np.concatenate(all_scores)
        })
    
    print(f"[{aspect}] Generated {len(df)} predictions")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate CAFA submission from trained Keras models"
    )
    parser.add_argument(
        "--model-c", required=True,
        help="Path to Keras model for Cellular Component (C) aspect"
    )
    parser.add_argument(
        "--model-f", required=True,
        help="Path to Keras model for Molecular Function (F) aspect"
    )
    parser.add_argument(
        "--model-p", required=True,
        help="Path to Keras model for Biological Process (P) aspect"
    )
    parser.add_argument(
        "--output", "-o", default="submission.tsv",
        help="Output path for submission TSV file (default: submission.tsv)"
    )
    parser.add_argument(
        "--embeddings", default="dataset/prott5_embs/test_t5_embs.parquet",
        help="Path to test embeddings parquet file"
    )
    parser.add_argument(
        "--embedding-col", default="embedding",
        help="Column name for embeddings in the parquet file"
    )
    parser.add_argument(
        "--taxon-topk", type=int, default=77,
        help="Number of top taxon IDs for one-hot encoding (default: 77)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Minimum score threshold for predictions (default: 0.01)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5000,
        help="Chunk size for memory-efficient prediction (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Validate model paths
    for path, aspect in [(args.model_c, "C"), (args.model_f, "F"), (args.model_p, "P")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model for aspect {aspect} not found: {path}")
    
    # Load label order from ontology
    print("Loading ontology and label order...")
    label_order_dict = get_label_order_dict()
    print(f"  C terms: {len(label_order_dict['C'])}")
    print(f"  F terms: {len(label_order_dict['F'])}")
    print(f"  P terms: {len(label_order_dict['P'])}")
    
    # Load test features
    X_test, protein_ids = load_test_features(
        embeddings_path=args.embeddings,
        embedding_col=args.embedding_col,
        taxon_topk=args.taxon_topk,
    )
    
    # Generate predictions for each aspect
    submission_c = generate_predictions_for_aspect(
        aspect="C",
        model_path=args.model_c,
        X_input=X_test,
        protein_ids=protein_ids,
        label_list=label_order_dict["C"],
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )
    
    submission_f = generate_predictions_for_aspect(
        aspect="F",
        model_path=args.model_f,
        X_input=X_test,
        protein_ids=protein_ids,
        label_list=label_order_dict["F"],
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )
    
    submission_p = generate_predictions_for_aspect(
        aspect="P",
        model_path=args.model_p,
        X_input=X_test,
        protein_ids=protein_ids,
        label_list=label_order_dict["P"],
        threshold=args.threshold,
        chunk_size=args.chunk_size,
    )
    
    # Combine all predictions
    print("\nCombining predictions from all aspects...")
    final_submission_df = pl.concat([submission_c, submission_f, submission_p])
    
    # Write to TSV (no header, tab-separated as per CAFA format)
    print(f"Saving submission to {args.output}...")
    final_submission_df.write_csv(args.output, separator="\t", include_header=False)
    
    print(f"\n✓ Submission saved with {len(final_submission_df):,} total predictions")
    print(f"  - C: {len(submission_c):,} predictions")
    print(f"  - F: {len(submission_f):,} predictions")
    print(f"  - P: {len(submission_p):,} predictions")


if __name__ == "__main__":
    main()

