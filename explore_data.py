import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from Bio import SeqIO
    import plotnine as p9
    import marimo as mo
    import torch
    import numpy as np
    return SeqIO, mo, np, p9, pl, torch


@app.cell
def _(SeqIO, pl):
    def create_train_df(fasta_path="dataset/Train/train_sequences.fasta", taxon_file="dataset/Train/train_taxonomy.tsv"):
        data = [
            (*record.id.split("|", 2), str(record.seq))
            for record in SeqIO.parse(fasta_path, "fasta")
        ]

        train_df = pl.DataFrame(
            data, 
            schema=["db", "protein_id", "gene_name", "sequence"],
            orient="row"
        )

        train_taxons_df = pl.read_csv(taxon_file, separator="\t", has_header=False).rename({"column_1": "protein_id", "column_2": "taxon_id"})

        return train_df.join(train_taxons_df, on="protein_id")

    train_df = create_train_df()
    train_df
    return (train_df,)


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
def _(pl, test_df, train_df):
    def one_hot_encode_taxa(df, reference_df=test_df, vec_len=16):
        taxon_counts = reference_df.group_by("taxon_id").len().sort("len", descending=True)
        top_taxons = taxon_counts[:vec_len]["taxon_id"].to_list()
        filtered_df = df.with_columns(
            pl.when(pl.col("taxon_id").is_in(top_taxons))
            .then(pl.col("taxon_id"))
            .otherwise(pl.lit(-1))
            .alias("taxon_id"))
        return filtered_df.with_columns(
            filtered_df.select("taxon_id").to_dummies()
        ).rename({"taxon_id_-1": "taxon_id_other"})

    train_one_hot_taxa_df = one_hot_encode_taxa(train_df)
    train_one_hot_taxa_df
    return one_hot_encode_taxa, train_one_hot_taxa_df


@app.cell
def _(one_hot_encode_taxa, test_df):
    test_one_hot_taxa_df = one_hot_encode_taxa(test_df)
    test_one_hot_taxa_df
    return (test_one_hot_taxa_df,)


@app.cell
def _(test_one_hot_taxa_df, train_one_hot_taxa_df):
    train_one_hot_taxa_df.write_parquet("dataset/Train/trainset_with_onehot_taxas.parquet", compression="zstd")
    test_one_hot_taxa_df.write_parquet("dataset/Test/testset_with_onehot_taxas.parquet", compression="zstd")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## BLAST Baseline
    """)
    return


@app.cell
def _(pl):
    train_terms_df = pl.read_csv("dataset/Train/train_terms.tsv", separator="\t")
    train_terms_df.head()
    return (train_terms_df,)


@app.cell
def _(pl, train_terms_df):
    # Define column names for BLAST format 6 output
    _blast_col_names = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen", 
        "qstart", "qend", "sstart", "send", "evalue", "bitscore"
    ]

    try:
        print("Reading BLAST results and computing max bit-score per term...")

        # Use Lazy API for efficient processing of potential large files
        # 1. Load BLAST results
        _blast_lf = pl.scan_csv(
            "results.txt", 
            separator="\t", 
            has_header=False, 
            new_columns=_blast_col_names
        )

        # 2. Clean sseqid to match EntryID format
        # Training FASTA headers (UniProt) are typically 'sp|Accession|ID'.
        # We extract 'Accession' (index 1) to join with EntryID in terms data.
        _blast_clean_lf = _blast_lf.with_columns(
            pl.when(pl.col("sseqid").str.contains(r"\|"))
            .then(pl.col("sseqid").str.split("|").list.get(1))
            .otherwise(pl.col("sseqid"))
            .alias("sseqid_clean")
        )

        # 3. Calculate total bitscore per query (denominator for knn_score)
        # This sums bitscores of all matched proteins for a query
        _blast_denom_lf = _blast_clean_lf.group_by("qseqid").agg(
            pl.col("bitscore").sum().alias("total_bitscore")
        )

        # 4. Join with Training Terms and Aggregate
        # We join hits to their terms using the filtered_terms_df from context
        # For each query (qseqid) and term, we calculate the max bitscore and the knn_score.
        blast_knn_preds_df = (
            _blast_clean_lf.join(
                train_terms_df.select(["EntryID", "term"]).lazy(),
                left_on="sseqid_clean",
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
            .collect()  # Execute the plan
            .sort(["qseqid", "knn_score"], descending=[False, True])
        )

        print(f"Success: Generated {blast_knn_preds_df.height} predictions.")

    except Exception as e:
        print(f"Could not process BLAST results: {e}")
        # Return empty dataframe to maintain notebook state
        blast_knn_preds_df = pl.DataFrame(schema={"qseqid": pl.Utf8, "term": pl.Utf8, "score": pl.Float64, "knn_score": pl.Float64})

    blast_knn_preds_df
    return (blast_knn_preds_df,)


@app.cell
def _(blast_knn_preds_df):
    blast_knn_preds_df.write_csv("blast_knn_preds_full.tsv", separator="\t", include_header=False);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ESM Embeddings
    """)
    return


@app.cell
def _(pl, test_one_hot_taxa_df, torch, tqdm):
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    def create_esmc_embeddings_from_df(df, esm_model="esmc_600m", device="cuda"):
        client = ESMC.from_pretrained(esm_model).to(device)
        client.eval()

        protein_ids = df["protein_id"].to_list()
        sequences = df["sequence"].to_list()
        embs = []

        with torch.inference_mode():
            for seq in tqdm(sequences, desc=f"Embedding with {esm_model}"):
                protein = ESMProtein(sequence=seq)
                protein_tensor = client.encode(protein)
                logits_output = client.logits(protein_tensor, LogitsConfig(sequence=False, return_embeddings=True))
                emb = logits_output.embeddings.mean(dim=1).squeeze()
                embs.append(emb.cpu().numpy())

        return pl.DataFrame({
            "protein_id": protein_ids,
            f"{esm_model}_embedding": embs, 
        })

    create_esmc_embeddings_from_df(test_one_hot_taxa_df, device="cuda:1").write_parquet("dataset/Test/test_esmc_600m.parquet", compression="zstd")
    return


@app.cell
def _(pl):
    train_esmc600_df = pl.read_parquet("dataset/Train/train_esmc_600m_labels.parquet")
    test_esmc600_df = pl.read_parquet("dataset/Test/test_esmc_600m.parquet")
    return test_esmc600_df, train_esmc600_df


@app.cell
def _(np, test_esmc600_df, train_esmc600_df):
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    train_emb_col = "esmc_600m_emb" if "esmc_600m_emb" in train_esmc600_df.columns else "embedding"
    test_emb_col = "esmc_600m_embedding"

    test_embs = np.stack(test_esmc600_df[test_emb_col].to_numpy())
    train_embs = np.stack(train_esmc600_df[train_emb_col].to_numpy())

    n_test = len(test_embs)
    batch_size = 512

    sims = []

    for i in tqdm(range(0, n_test, batch_size)):
        if i < 1:
            batch_end = min(i + batch_size, n_test)
            X_batch = test_embs[i:batch_end] 
            sims.append(cosine_similarity(X_batch, train_embs))
        else:
            break

    sims
    return (tqdm,)


@app.cell
def _(np, pl, test_esmc600_df, torch, tqdm, train_esmc600_df, train_terms_df):
    import torch.nn.functional as F
    import os
    import shutil
    import glob
    import gc

    def get_embedding_knn_preds(
        test_df, 
        train_df, 
        terms_df, 
        test_emb_col, 
        train_emb_col,
        cutoff=0.85, 
        topk=50,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=512
    ):
        """
        Computes KNN-based scores using Embedding Cosine Similarity.
        Score = Sum(Sim * Label) / Sum(Sim) for neighbors where Sim > cutoff.
        Writes intermediate batches to disk to reduce memory usage.
        """
        print(f"Computing Embedding KNN on {device} with cutoff={cutoff} and topk={topk}...")

        # Setup temp directory for batch swamping
        temp_dir = "temp_knn_batches"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        # 1. Setup Terms Map
        # Convert terms to integer indices for sparse matrix construction
        unique_terms = terms_df["term"].unique().sort()
        term_to_idx = {t: i for i, t in enumerate(unique_terms.to_list())}
        idx_to_term = {i: t for t, i in term_to_idx.items()}
        n_terms = len(unique_terms)

        # 2. Prepare Train Tensors (X_train)
        # Normalize embeddings for Cosine Similarity
        # Ensure embeddings are a numpy stack (handle if list series)
        train_embs = np.stack(train_df[train_emb_col].to_numpy())
        X_train = torch.tensor(train_embs, dtype=torch.float32, device=device)
        X_train = F.normalize(X_train, p=2, dim=1)
        n_train = X_train.shape[0]

        # 3. Prepare Labels Transposed (Y_T) for efficient Multiplication
        # Shape: (N_terms, N_train)
        print("Mapping annotations to sparse tensor...")
        pid_to_idx = {pid: i for i, pid in enumerate(train_df["protein_id"].to_list())}

        # Filter terms for proteins present in our embedding set
        target_ids = set(train_df["protein_id"])

        # Map protein IDs and Terms to indices
        # Use Polars for speed
        term_map_df = (
            terms_df
            .filter(pl.col("EntryID").is_in(target_ids))
            .select(["EntryID", "term"])
            .with_columns([
                pl.col("EntryID").replace(pid_to_idx, default=None).cast(pl.Int64).alias("row_idx"),
                pl.col("term").replace(term_to_idx, default=None).cast(pl.Int64).alias("col_idx")
            ])
            .drop_nulls()
        )

        # Create Sparse Tensor Y.T (Rows=Terms, Cols=Proteins)
        # We transpose logically here to support torch.sparse.mm(Sparse, Dense)
        indices = torch.tensor(
            np.stack([term_map_df["col_idx"].to_numpy(), term_map_df["row_idx"].to_numpy()]), 
            dtype=torch.long,
            device=device
        )
        values = torch.ones(term_map_df.height, dtype=torch.float32, device=device)
        Y_T = torch.sparse_coo_tensor(indices, values, (n_terms, n_train))

        # 4. Process Test Data in Batches
        test_ids = test_df["protein_id"].to_list()
        test_embs = np.stack(test_df[test_emb_col].to_numpy())
        X_test_all = torch.tensor(test_embs, dtype=torch.float32, device=device)
        X_test_all = F.normalize(X_test_all, p=2, dim=1)

        n_test = len(test_ids)

        print(f"Running inference on {n_test} sequences...")

    
        # Accumulate local results for this batch
        batch_pids = []
        batch_terms = []
        batch_scores = []

        for i in tqdm(range(0, n_test, batch_size)):
            batch_end = min(i + batch_size, n_test)
            # Slice Test Batch (B, D)
            X_batch = X_test_all[i:batch_end] 
            current_batch_size = X_batch.shape[0]

            # Compute Similarity: (B, D) @ (D, N_train) -> (B, N_train)
            sim_batch = X_batch @ X_train.T

            # --- FILTERING STEP TO REDUCE OUTPUT SIZE ---
            # 1. Keep only Top-K neighbors per test protein
            # 2. Apply Cutoff threshold
            k_val = min(topk, n_train)
            top_vals, top_inds = torch.topk(sim_batch, k=k_val, dim=1)
        
            # Apply cutoff: zero out low similarity neighbors
            top_vals = torch.where(top_vals > cutoff, top_vals, torch.tensor(0.0, device=device))
        
            # Reconstruct sparse-like sim matrix (reuse memory)
            # Zero out the full density matrix, then scatter the top-k values back
            sim_batch.zero_().scatter_(1, top_inds, top_vals)

            # Transpose for matrix multiplication: (N_train, B)
            sim_batch_T = sim_batch.T 

            # Calculate Denominator (Sum of similarities per test protein)
            # Sum over N_train (dim 0) -> (B, )
            denom = sim_batch_T.sum(dim=0)

            # Calculate Numerator
            # Y_T (N_terms, N_train) @ sim_batch_T (N_train, B) -> (N_terms, B)
            # This effectively sums the similarity scores for each term
            num = torch.sparse.mm(Y_T, sim_batch_T)

            # Compute Scores (N_terms, B)
            scores = num / (denom.unsqueeze(0) + 1e-9)

            # Extract relevant scores to CPU
            scores_cpu = scores.cpu()


            # Parse results for this batch
            for local_idx in range(current_batch_size):
                p_id = test_ids[i + local_idx]
                p_scores = scores_cpu[:, local_idx]

                # Keep only non-zero scores to save memory/time
                valid_mask = p_scores > 0
                if valid_mask.any():
                    valid_indices = torch.nonzero(valid_mask).squeeze(1).numpy()
                    valid_vals = p_scores[valid_indices].numpy()

                    batch_terms_list = [idx_to_term[ix] for ix in valid_indices]

                    batch_pids.extend([p_id] * len(batch_terms_list))
                    batch_terms.extend(batch_terms_list)
                    batch_scores.extend(valid_vals)

        return pl.DataFrame({
                    "protein_id": batch_pids,
                    "term": batch_terms,
                    "knn_score": batch_scores
                }, schema={"protein_id": pl.Utf8, "term": pl.Utf8, "knn_score": pl.Float32})

    

    # --- Execution ---
    # Detect correct column names based on prior context
    _train_emb_col = "esmc_600m_emb" if "esmc_600m_emb" in train_esmc600_df.columns else "embedding"
    _test_emb_col = "esmc_600m_embedding"

    # Compute predictions
    emb_knn_preds_df = get_embedding_knn_preds(
        test_esmc600_df,
        train_esmc600_df,
        train_terms_df,
        test_emb_col=_test_emb_col,
        train_emb_col=_train_emb_col,
        cutoff=0.95, # Tunable: 0.8 to 0.9 usually good for ESM
        topk=10      # Limit to top 50 neighbors to prevent explosion
    )

    # Save and display
    emb_knn_preds_df.write_csv("results/esmc600_knn_preds.tsv", separator="\t", include_header=False)
    print(f"Generated {emb_knn_preds_df.height} predictions.")
    emb_knn_preds_df.head(10)
    return (emb_knn_preds_df,)


@app.cell
def _(emb_knn_preds_df):
    emb_knn_preds_df.write_csv("esmc600_knn_preds.tsv", separator="\t", include_header=False)
    return


@app.cell
def _(pl):
    final_df = pl.scan_parquet("temp_knn_batches/*.parquet")
    return (final_df,)


@app.cell
def _(final_df):
    final_df.head().collect()
    return


@app.cell
def _(final_df):
    final_df.sink_csv("esmc600_knn_preds.tsv", separator="\t", include_header=False)
    return


@app.cell
def _(pl):
    pl.read_parquet("temp_knn_batches/batch_0.parquet")["protein_id", "term"].n_unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cafa Eval
    """)
    return


@app.cell
def _():
    from cafaeval.evaluation import cafa_eval, write_results

    df1, dfs_best = cafa_eval(
      obo_file="dataset/Train/go-basic.obo",
      pred_dir="CAFA-evaluator-PK/example/predictions",
      gt_file="CAFA-evaluator-PK/example/ground_truth_limited.tsv",
      th_step=0.01,
      n_cpu=4,
    )

    write_results(df1, dfs_best, out_dir="results", th_step=0.01)
    return


if __name__ == "__main__":
    app.run()
