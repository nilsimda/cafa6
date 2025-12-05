import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import joblib
    from utils import data
    from dotenv import load_dotenv
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    from cafaeval.graph import propagate, Prediction
    from cafaeval.evaluation import evaluate_prediction
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import keras
    import wandb
    from wandb.integration.keras import WandbMetricsLogger
    import altair as alt
    return (
        MultiLabelBinarizer,
        Prediction,
        WandbMetricsLogger,
        alt,
        data,
        evaluate_prediction,
        keras,
        mo,
        np,
        pl,
        propagate,
        wandb,
    )


@app.cell
def _(data, pl):
    train_df = data.create_train_df()
    train_embs_df = pl.read_parquet("dataset/Train/esmc_600m_embs.parquet")
    return train_df, train_embs_df


@app.cell
def _(data):
    test_df = data.create_test_df()
    test_df
    return (test_df,)


@app.cell
def _(pl, test_df, train_df):
    top_ids = test_df.group_by("taxon_id").len().sort("len", descending=True).head(20).get_column("taxon_id").to_list()
    phylum_map = pl.read_csv("dataset/taxon_phylum.tsv", separator="\t", has_header=False).rename({"column_1": "taxon_id", "column_2": "phylum"}).with_columns(pl.col("taxon_id").cast(pl.Int64))

    train_taxon_cat_df = train_df.with_columns(pl.col("taxon_id").cast(pl.Int64)).join(phylum_map, on="taxon_id", how="left").with_columns(
        pl.when(pl.col("taxon_id").is_in(top_ids))
          .then(pl.col("taxon_id").cast(pl.String))
          .otherwise(pl.col("phylum").fill_null("Other"))
          .alias("taxon_cat")
    )

    test_taxon_cat_df = test_df.with_columns(pl.col("taxon_id").cast(pl.Int64)).join(phylum_map, on="taxon_id", how="left").with_columns(
        pl.when(pl.col("taxon_id").is_in(top_ids))
          .then(pl.col("taxon_id").cast(pl.String))
          .otherwise(pl.col("phylum").fill_null("Other"))
          .alias("taxon_cat")
    )
    return test_taxon_cat_df, train_taxon_cat_df


@app.cell
def _(alt, mo, test_taxon_cat_df, train_taxon_cat_df):
    train_taxon_cat_ss_df = train_taxon_cat_df.sample(10_000)
    test_taxon_cat_ss_df = test_taxon_cat_df.sample(10_000)

    max_y = max(
        train_taxon_cat_ss_df["taxon_cat"].value_counts().max().get_column("count").item(),
        test_taxon_cat_ss_df["taxon_cat"].value_counts().max().get_column("count").item()
    )

    y_enc = alt.Y("count()", scale=alt.Scale(domain=[0, max_y]))

    chart1 = alt.Chart(train_taxon_cat_ss_df).mark_bar().encode(
        y=y_enc, x="taxon_cat"
    )

    chart2 = alt.Chart(test_taxon_cat_ss_df).mark_bar().encode(
        y=y_enc, x="taxon_cat"
    )

    mo.ui.altair_chart(chart1 & chart2)
    return


@app.cell
def _(pl):
    test_embs_df = pl.read_parquet("dataset/Test/test_esmc_600m.parquet")
    test_embs_df
    return (test_embs_df,)


@app.cell
def _(data, test_df, train_df):
    train_ohe_df = data.onehot_encode_taxon_ids(train_df, k=20)
    test_ohe_df = data.onehot_encode_taxon_ids(test_df, k=20)
    return test_ohe_df, train_ohe_df


@app.cell
def _(train_embs_df, train_ohe_df):
    train_ohe_embs_df = train_ohe_df.join(train_embs_df, on="protein_id", how="inner")
    train_ohe_embs_df
    return (train_ohe_embs_df,)


@app.cell
def _(test_embs_df, test_ohe_df):
    test_ohe_embs_df = test_ohe_df.join(test_embs_df, on="protein_id", how="inner")
    test_ohe_embs_df
    return (test_ohe_embs_df,)


@app.cell
def _(pl):
    def create_stratified_split(df, test_df, category_col, valid_size=0.2):
        """
        Create a validation set that matches the test set's categorical distribution.

        Parameters:
        - df: Training dataframe to split
        - test_df: Test dataframe (used only to compute target distribution)
        - category_col: Name of categorical column to stratify on
        - valid_size: Fraction of df to use for validation
        """

        # Get the distribution from test set
        test_dist = (
            test_df
            .group_by(category_col)
            .agg(pl.len().alias('count'))
            .with_columns((pl.col('count') / pl.col('count').sum()).alias('proportion'))
        )

        # Sample from each category proportionally
        valid_dfs = []
        train_dfs = []

        for row in test_dist.iter_rows(named=True):
            category = row[category_col]
            target_prop = row['proportion']

            # Get all rows for this category
            category_df = df.filter(pl.col(category_col) == category)
            n_total = len(category_df)

            # Calculate how many to sample for validation
            n_valid = int(n_total * valid_size * target_prop / 
                          (test_dist.filter(pl.col(category_col) == category)['proportion'][0]))
            n_valid = min(n_valid, n_total)  # Don't exceed available samples

            # Sample
            valid_sample = category_df.sample(n=n_valid, shuffle=True, seed=42)
            train_sample = category_df.join(valid_sample, how='anti', on=df.columns)

            valid_dfs.append(valid_sample)
            train_dfs.append(train_sample)

        valid_set = pl.concat(valid_dfs)
        train_set = pl.concat(train_dfs)

        return train_set, valid_set

    # Usage
    #train_df, valid_df = create_stratified_split(df, test_df, 'category_column', valid_size=0.2)
    return (create_stratified_split,)


@app.cell
def _(create_stratified_split, pl, test_taxon_cat_df, train_taxon_cat_df):
    def create_trainval_split():
        labels_df = pl.read_csv("dataset/Train/train_terms.tsv", separator="\t")
        aspect_dfs = labels_df.group_by("EntryID", "aspect").agg(pl.col("term")).partition_by(by="aspect", as_dict=True)


        for aspect in aspect_dfs.keys():
            aspect_labels_df = aspect_dfs[aspect]
            aspect_full_df = aspect_labels_df.join(train_taxon_cat_df, how="left", left_on="EntryID", right_on="protein_id")
            trainset_df_aspect, valset_df_aspect = create_stratified_split(aspect_full_df, test_taxon_cat_df, "taxon_cat", valid_size=0.2)
            trainset_df_aspect["EntryID", "term", "aspect"].explode("term").write_csv(f"dataset/Split/train_aspect_{aspect[0]}_gt.tsv", separator="\t")
            valset_df_aspect["EntryID", "term", "aspect"].explode("term").write_csv(f"dataset/Split/val_aspect_{aspect[0]}_gt.tsv", separator="\t")

    create_trainval_split()
    return


@app.cell
def _():
    from cafaeval.parser import obo_parser, gt_parser

    ontologies = obo_parser("dataset/Train/go-basic.obo", ("is_a", "part_of"),  "dataset/IA.tsv", True)

    gts = {aspect: gt_parser(f"dataset/Split/val_aspect_{aspect}_gt.tsv", ontologies) for aspect in ["C", "F", "P"]}

    label_order_c = [term["id"] for term in ontologies["cellular_component"].terms_list]
    label_order_f = [term["id"] for term in ontologies["molecular_function"].terms_list]
    label_order_p = [term["id"] for term in ontologies["biological_process"].terms_list]

    label_order_dict = {"C": label_order_c, "F": label_order_f, "P": label_order_p}
    return gts, label_order_dict, ontologies


@app.cell
def _(MultiLabelBinarizer, label_order_dict, np, pl, train_ohe_embs_df):
    def create_aspect_dataset(aspect):
        train_aspect_terms_df = pl.read_csv(f"dataset/Split/train_aspect_{aspect}_gt.tsv", separator="\t")
        val_aspect_terms_df = pl.read_csv(f"dataset/Split/val_aspect_{aspect}_gt.tsv", separator="\t")

        train_aspect_df = train_aspect_terms_df.group_by("EntryID").agg(pl.col("term")).join(train_ohe_embs_df, how="left", left_on="EntryID", right_on="protein_id")

        val_aspect_df = val_aspect_terms_df.group_by("EntryID").agg(pl.col("term")).join(train_ohe_embs_df, how="left", left_on="EntryID", right_on="protein_id")

        train_taxon_ohe_aspect = np.stack(train_aspect_df["taxon_ohe"].to_list())
        train_embs_aspect = np.stack(train_aspect_df["embedding"].to_list())
        X_train = np.hstack([train_taxon_ohe_aspect, train_embs_aspect])

        val_taxon_ohe_aspect = np.stack(val_aspect_df["taxon_ohe"].to_list())
        val_embs_aspect = np.stack(val_aspect_df["embedding"].to_list())
        X_val = np.hstack([val_taxon_ohe_aspect, val_embs_aspect])

        mlb = MultiLabelBinarizer(classes=label_order_dict[aspect])
        mlb.fit(train_aspect_df.vstack(val_aspect_df)["term"])

        y_train = mlb.transform(train_aspect_df["term"])
        y_val = mlb.transform(val_aspect_df["term"])


        val_prot_ids_aspect = val_aspect_df["EntryID"].to_list()
        return X_train, X_val, y_train, y_val, val_prot_ids_aspect

    X_train_c, X_val_c, y_train_c, y_val_c, val_prot_ids_c = create_aspect_dataset("C")
    X_train_f, X_val_f, y_train_f, y_val_f, val_prot_ids_f = create_aspect_dataset("F")
    X_train_p, X_val_p, y_train_p, y_val_p, val_prot_ids_p = create_aspect_dataset("P")
    return (
        X_train_c,
        X_train_f,
        X_train_p,
        X_val_c,
        X_val_f,
        X_val_p,
        val_prot_ids_c,
        val_prot_ids_f,
        val_prot_ids_p,
        y_train_c,
        y_train_f,
        y_train_p,
        y_val_c,
        y_val_f,
        y_val_p,
    )


@app.cell
def _(Prediction, evaluate_prediction, keras, np, propagate):
    class CafaEvalCallback(keras.callbacks.Callback):
        def __init__(self, validation_data, prot_ids, ns, ontologies, gt):
            super().__init__()
            self.X_val, self.y_val = validation_data
            self.prot_ids = prot_ids
            self.ns = ns
            self.ontologies = ontologies
            self.gt = gt
            self.tau_arr = np.arange(0.1, 1, 0.1)

        def on_epoch_end(self, epoch, logs=None):
            # Get predictions for all validation data
            y_pred = self.model.predict(self.X_val, verbose=0)

            gt_ns = self.gt[self.ns]
            num_prots, num_terms = gt_ns.matrix.shape
            mat = np.zeros((num_prots, num_terms), dtype=np.float32)

            for prot_id, scores in zip(self.prot_ids, y_pred):
                if prot_id in gt_ns.ids:
                    mat[gt_ns.ids[prot_id]] = scores

            propagate(mat, self.ontologies[self.ns], 
                     self.ontologies[self.ns].order, mode='max')
            predictions = {self.ns: Prediction(gt_ns.ids, mat, self.ns)}

            eval_df = evaluate_prediction(predictions, self.gt, self.ontologies,
                                         self.tau_arr, normalization="cafa", n_cpu=1)

            rc_w_score = eval_df["rc_w"].max().item()
            pr_w_score = eval_df["pr_w"].max().item()
            f_w_score = eval_df["f_w"].max().item()
            f_micro_w_score = eval_df["f_micro_w"].max().item()

            logs['val_cafaeval_f_w'] = f_w_score
            logs['val_cafaeval_f_micro_w'] = f_micro_w_score
            logs['val_cafaeval_rc_w'] = rc_w_score
            logs['val_cafaeval_pr_w'] = pr_w_score
    return (CafaEvalCallback,)


@app.cell
def _(keras):
    def build_mlp_aspect_model(input_dim, output_dim):
        """
        Builds a Multilayer Perceptron for multi-label classification.
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(2048, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(output_dim, activation="sigmoid")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
        )
        return model
    return (build_mlp_aspect_model,)


@app.cell
def _(label_order_dict, pl):
    IA_df = pl.read_csv("dataset/IA.tsv", separator="\t", has_header=False)
    class_weights_by_aspect = {}
    for aspect in ["C", "F", "P"]:
        ia_w = IA_df.filter(pl.col("column_1").is_in(label_order_dict[aspect])).sort(by="column_1")["column_2"].to_list()
        weights_dict = {i: w for i, w in enumerate(ia_w)}
        class_weights_by_aspect[aspect] = weights_dict

    class_weights_by_aspect
    return (class_weights_by_aspect,)


@app.cell
def _(
    CafaEvalCallback,
    WandbMetricsLogger,
    X_train_c,
    X_train_f,
    X_train_p,
    X_val_c,
    X_val_f,
    X_val_p,
    batch_size,
    build_mlp_aspect_model,
    class_weights_by_aspect,
    epochs,
    gts,
    keras,
    ontologies,
    val_prot_ids_c,
    val_prot_ids_f,
    val_prot_ids_p,
    wandb,
    y_train_c,
    y_train_f,
    y_train_p,
    y_val_c,
    y_val_f,
    y_val_p,
):
    def train_aspect_model(
        aspect,
        X_train,
        y_train,
        X_val,
        y_val,
        val_prot_ids,
        ontologies,
        gts,
        epochs=500,
        batch_size=512,
        learning_rate=0.01,
        wandb_project="cafa-6",
        wandb_name_suffix="_decay_lr_on_plat"
    ):
        """
        Trains and evaluates an MLP model for a specific GO aspect (C, F, or P).
        """
        namespace_map = {
            "C": "cellular_component",
            "F": "molecular_function",
            "P": "biological_process"
        }

        # Identify namespace and Ground Truth for this aspect
        ns = namespace_map[aspect]
        gt_aspect = gts[aspect]

        # Initialize WandB run
        run_name = f"MLP_Aspect_{aspect}{wandb_name_suffix}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "architecture": "MLP",
                "aspect": aspect,
                "input_dim": X_train.shape[1],
                "output_dim": y_train.shape[1],
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
            },
        )

        print(f"[{aspect}] Building model...")
        # Build model using function from previous context
        model = build_mlp_aspect_model(X_train.shape[1], y_train.shape[1])

        # Re-compile to ensure specific learning rate is used
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy"
        )

        # Define Callbacks
        model_save_path = f"keras_models/best_mlp_model_{aspect.lower()}.keras"
        callbacks = [
            CafaEvalCallback(
                (X_val, y_val), 
                val_prot_ids, 
                ns, 
                ontologies, 
                gt_aspect
            ),
            #keras.callbacks.EarlyStopping(monitor="val_cafaeval_f_w", patience=10, mode="max"),
            keras.callbacks.ReduceLROnPlateau(monitor="val_cafaeval_f_w", mode="max"),
            keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor="val_cafaeval_f_w",
                mode="max",
                save_best_only=True,
                verbose=1
            ),
            WandbMetricsLogger()
        ]

        print(f"[{aspect}] Starting training...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights_by_aspect[aspect]
        )

        wandb.finish()
        print(f"[{aspect}] Training finished. Best model saved to {model_save_path}")

        return model, history

    # Execute training for all three aspects using the function
    _model_c, _history_c = train_aspect_model(
        "C", X_train_c, y_train_c, X_val_c, y_val_c, val_prot_ids_c, ontologies, gts)

    _model_f, _history_f = train_aspect_model(
        "F", X_train_f, y_train_f, X_val_f, y_val_f, val_prot_ids_f, ontologies, gts, 
        epochs=epochs, batch_size=batch_size
    )

    _model_p, _history_p = train_aspect_model(
        "P", X_train_p, y_train_p, X_val_p, y_val_p, val_prot_ids_p, ontologies, gts, epochs=300
    )
    return


@app.cell
def _(np, test_ohe_embs_df):
    _test_ohes = np.stack(test_ohe_embs_df["taxon_ohe"].to_list())
    _test_embs = np.stack(test_ohe_embs_df["esmc_600m_embedding"].to_list())
    X_test = np.hstack([_test_ohes, _test_embs])
    return (X_test,)


@app.cell
def _(X_test, keras, label_order_dict, np, pl, test_ohe_embs_df):
    import gc
    from tqdm import tqdm

    def generate_predictions_for_aspect(aspect, model_path, X_input, protein_ids, label_list, threshold=0.01, chunk_size=5000):
        """
        Generates predictions for a specific aspect using a saved Keras model,
        optimizing for memory by processing in chunks and clearing objects immediately.
        """
        print(f"Loading model for aspect {aspect} from {model_path}...")
        model = keras.models.load_model(model_path)

        # Convert label list to array for fast indexing
        term_arr = np.array(label_list)

        all_ids = []
        all_terms = []
        all_scores = []

        num_samples = len(protein_ids)
        print(f"Generating predictions for {aspect} in chunks...")

        # Iterate in chunks to avoid OOM
        for start_idx in tqdm(range(0, num_samples, chunk_size)):
            end_idx = min(start_idx + chunk_size, num_samples)

            # Prepare batch
            X_chunk = X_input[start_idx:end_idx]
            ids_chunk = protein_ids[start_idx:end_idx]

            # Predict
            # Using a proper batch_size for GPU efficiency, while chunk_size manages RAM
            y_pred = model.predict(X_chunk, verbose=0, batch_size=128)

            # Filter predictions
            _rows, _cols = np.where(y_pred > threshold)
            _scores = y_pred[_rows, _cols]

            # Append valid predictions
            if len(_scores) > 0:
                all_ids.append(ids_chunk[_rows])
                all_terms.append(term_arr[_cols])
                all_scores.append(_scores)

            # Clean up chunk memory
            del y_pred, _rows, _cols, _scores, X_chunk
            gc.collect()

        # Explicit memory cleanup
        del model
        gc.collect()

        print(f"Concatenating results for {aspect}...")
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

        return df

    # Get test IDs once
    _test_protein_ids = test_ohe_embs_df["protein_id"].to_numpy()

    # Generate predictions for each aspect
    submission_c = generate_predictions_for_aspect(
        aspect="C",
        model_path="keras_models/best_mlp_model_c.keras",
        X_input=X_test,
        protein_ids=_test_protein_ids,
        label_list=label_order_dict["C"]
    )

    submission_f = generate_predictions_for_aspect(
        aspect="F",
        model_path="keras_models/best_mlp_model_f.keras",
        X_input=X_test,
        protein_ids=_test_protein_ids,
        label_list=label_order_dict["F"]
    )

    submission_p = generate_predictions_for_aspect(
        aspect="P",
        model_path="keras_models/best_mlp_model_p.keras",
        X_input=X_test,
        protein_ids=_test_protein_ids,
        label_list=label_order_dict["P"]
    )

    # Combine all predictions
    print("Concatenating and saving submission...")
    final_submission_df = pl.concat([submission_c, submission_f, submission_p])

    # Write to TSV (no header, tab-separated as per CAFA format)
    final_submission_df.write_csv("submission.tsv", separator="\t", include_header=False)
    print(f"Submission saved with {len(final_submission_df)} predictions.")

    final_submission_df
    return


@app.cell
def _(input_dim, keras, output_dim):
    import keras_tuner as kt

    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_dim,)))
    
        # Search over 1-3 layers
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(keras.layers.Dense(
                hp.Choice(f'units_{i}', [256, 512, 1024, 2048]),
                activation='relu'
            ))
        
            # Either BN or Dropout, not both
            if hp.Boolean('use_batchnorm'):
                model.add(keras.layers.BatchNormalization())
            else:
                model.add(keras.layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
    
        model.add(keras.layers.Dense(output_dim, activation='sigmoid'))
    
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3])
            ),
            loss='binary_crossentropy'
        )
        return model

    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective('val_cafaeval_f_w', direction='max'),
        max_trials=30,
        directory='kt_search',
        project_name='cafa6_aspect_c'
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
