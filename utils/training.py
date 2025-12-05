"""
Training utilities for CAFA-6 MLP models.

This module provides reusable components for training GO term prediction models:
- ModelConfig: Configuration dataclass for model architecture and training
- focal_loss: Focal loss for handling class imbalance
- build_model: Model construction from config
- CafaEvalCallback: CAFA evaluation during training
- load_data_for_aspect: Data loading pipeline
- train_model: High-level training function
"""

import os

# Set backend before importing keras
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import polars as pl
from sklearn.preprocessing import MultiLabelBinarizer
from dataclasses import dataclass, field
from typing import Callable

from utils import data
from cafaeval.parser import obo_parser, gt_parser
from cafaeval.graph import propagate, Prediction
from cafaeval.evaluation import evaluate_prediction


# ============== Configuration ==============

@dataclass
class ModelConfig:
    """Configuration for MLP model architecture and training.
    
    Attributes:
        name: Identifier for this configuration
        num_layers: Number of hidden layers
        layer_sizes: Tuple of units per layer
        learning_rate: Initial learning rate for Adam optimizer
        dropout: Dropout rate (if use_dropout=True)
        use_batchnorm: Whether to use batch normalization
        use_dropout: Whether to use dropout
        batch_size: Training batch size
        epochs: Maximum training epochs
        use_focal_loss: Whether to use focal loss instead of BCE
        focal_gamma: Gamma parameter for focal loss
        l2_reg: L2 regularization strength (0 = no regularization)
        taxon_topk: Number of top taxon IDs for one-hot encoding (0 = no taxon features)
    """
    name: str = "default"
    num_layers: int = 2
    layer_sizes: tuple = (1024, 512)
    learning_rate: float = 0.001
    dropout: float = 0.3
    use_batchnorm: bool = True
    use_dropout: bool = True
    batch_size: int = 512
    epochs: int = 30
    use_focal_loss: bool = False
    l2_reg: float = 0.0
    taxon_topk: int = 77



# ============== Model Building ==============

def build_model(config: ModelConfig, input_dim: int, output_dim: int) -> keras.Model:
    """Build MLP model based on configuration.
    
    Args:
        config: Model configuration
        input_dim: Number of input features
        output_dim: Number of output classes (GO terms)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    
    regularizer = keras.regularizers.L2(config.l2_reg) if config.l2_reg > 0 else None
    
    for i, units in enumerate(config.layer_sizes[:config.num_layers]):
        model.add(keras.layers.Dense(units, activation="relu", kernel_regularizer=regularizer))
        
        if config.use_batchnorm:
            model.add(keras.layers.BatchNormalization())
        
        if config.use_dropout:
            model.add(keras.layers.Dropout(config.dropout))
    
    model.add(keras.layers.Dense(output_dim, activation="sigmoid"))
    
    # Choose loss
    loss_fn = "binary_focal_crossentropy" if config.use_focal_loss else "binary_crossentropy"
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=loss_fn
    )
    
    return model


# ============== Callbacks ==============

class CafaEvalCallback(keras.callbacks.Callback):
    """Callback for computing CAFA evaluation metrics during training.
    
    Computes weighted F-max, precision, and recall using the official
    CAFA evaluation methodology with graph-aware score propagation.
    
    Attributes:
        best_f_w: Best weighted F-max score seen during training
    """
    
    def __init__(self, validation_data, prot_ids, ns, ontologies, gt):
        """Initialize the callback.
        
        Args:
            validation_data: Tuple of (X_val, y_val)
            prot_ids: List of protein IDs corresponding to validation samples
            ns: Namespace string (e.g., "cellular_component")
            ontologies: Dictionary of parsed ontologies
            gt: Ground truth dictionary for the aspect
        """
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.prot_ids = prot_ids
        self.ns = ns
        self.ontologies = ontologies
        self.gt = gt
        self.tau_arr = np.arange(0.1, 1, 0.1)
        self.best_f_w = 0.0

    def on_epoch_end(self, epoch, logs=None):
        assert self.model is not None, "Model is not initialized"
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
        
        f_w_score = eval_df["f_w"].max().item()
        best_idx = eval_df["f_w"].idxmax()
        
        if logs is not None:
            logs['val_cafaeval_f_w'] = f_w_score
            logs['val_cafaeval_rc_w'] = eval_df[best_idx, "rc_w"]
            logs['val_cafaeval_pr_w'] = eval_df[best_idx, "pr_w"]
        
        if f_w_score > self.best_f_w:
            self.best_f_w = f_w_score


# ============== Data Loading ==============

def load_data_for_aspect(
    aspect: str,
    taxon_topk: int = 77,
    embeddings_path: str = "dataset/prott5_embs/train_t5_embs.parquet",
    ontology_path: str = "dataset/Train/go-basic.obo",
    ia_path: str = "dataset/IA.tsv",
) -> tuple:
    """Load training and validation data for a specific GO aspect.
    
    Args:
        aspect: GO aspect ("C", "F", or "P")
        taxon_topk: Number of top taxon IDs for one-hot encoding (0 = no taxon features)
        embeddings_path: Path to protein embeddings parquet file
        ontology_path: Path to GO ontology OBO file
        ia_path: Path to information accretion file
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, val_prot_ids, ontologies, gts, ns)
        - X_train, X_val: Feature matrices (embeddings + taxon OHE)
        - y_train, y_val: Multi-label binary matrices
        - val_prot_ids: List of validation protein IDs
        - ontologies: Parsed ontology dictionary
        - gts: Ground truth dictionary
        - ns: Namespace string for the aspect
    """
    # Load ontologies and ground truth
    ontologies = obo_parser(ontology_path, ("is_a", "part_of"), ia_path, True)
    gts = {aspect: gt_parser(f"dataset/Split/val_aspect_{aspect}_gt.tsv", ontologies)}
    
    # Get label order
    ns_map = {"C": "cellular_component", "F": "molecular_function", "P": "biological_process"}
    label_order = [term["id"] for term in ontologies[ns_map[aspect]].terms_list]
    
    # Load embeddings
    train_embs_df = pl.read_parquet(embeddings_path)
    train_df = data.create_train_df()
    
    # Support no taxon ohe (k=0)—just use embeddings
    if taxon_topk > 0:
        train_ohe_df = data.onehot_encode_taxon_ids(train_df, k=taxon_topk)
        train_ohe_embs_df = train_ohe_df.join(train_embs_df, on="protein_id", how="inner")
    else:
        # No taxon features - just embeddings with empty OHE
        train_ohe_embs_df = train_embs_df.with_columns([
            pl.lit([]).alias("taxon_ohe")
        ]).select(["protein_id", "embedding", "taxon_ohe"])

    # Load aspect-specific data
    train_terms_df = pl.read_csv(f"dataset/Split/train_aspect_{aspect}_gt.tsv", separator="\t")
    val_terms_df = pl.read_csv(f"dataset/Split/val_aspect_{aspect}_gt.tsv", separator="\t")

    train_aspect_df = train_terms_df.group_by("EntryID").agg(pl.col("term")).join(
        train_ohe_embs_df, how="inner", left_on="EntryID", right_on="protein_id"
    )
    val_aspect_df = val_terms_df.group_by("EntryID").agg(pl.col("term")).join(
        train_ohe_embs_df, how="inner", left_on="EntryID", right_on="protein_id"
    )
    
    # Create feature matrices
    if taxon_topk > 0:
        train_taxon_ohe = np.stack(train_aspect_df["taxon_ohe"].to_list())
        val_taxon_ohe = np.stack(val_aspect_df["taxon_ohe"].to_list())
    else:
        train_taxon_ohe = np.zeros((train_aspect_df.height, 0), dtype=np.float32)
        val_taxon_ohe = np.zeros((val_aspect_df.height, 0), dtype=np.float32)
    
    train_embs = np.stack(train_aspect_df["embedding"].to_list())
    X_train = np.hstack([train_taxon_ohe, train_embs])
    val_embs = np.stack(val_aspect_df["embedding"].to_list())
    X_val = np.hstack([val_taxon_ohe, val_embs])
    
    # Create label matrices
    mlb = MultiLabelBinarizer(classes=label_order)
    mlb.fit(train_aspect_df.vstack(val_aspect_df)["term"])
    y_train = mlb.transform(train_aspect_df["term"])
    y_val = mlb.transform(val_aspect_df["term"])
    
    val_prot_ids = val_aspect_df["EntryID"].to_list()
    
    return X_train, X_val, y_train, y_val, val_prot_ids, ontologies, gts, ns_map[aspect]


# ============== High-Level Training ==============

def train_model(
    config: ModelConfig,
    aspect: str = "C",
    use_wandb: bool = False,
    wandb_project: str = "cafa-6",
    early_stopping_patience: int = 20,
    reduce_lr_patience: int = 8,
    save_path: str | None = None,
    verbose: int = 1,
) -> tuple[keras.Model, float, dict]:
    """Train a model with the given configuration.
    
    This is a high-level function that handles the full training pipeline:
    data loading, model building, callback setup, training, and optionally saving.
    
    Args:
        config: Model configuration
        aspect: GO aspect ("C", "F", or "P")
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name (if use_wandb=True)
        early_stopping_patience: Patience for early stopping
        reduce_lr_patience: Patience for learning rate reduction
        save_path: Path to save the trained model (None = don't save)
        verbose: Verbosity level for training
    
    Returns:
        Tuple of (model, best_f_w, history_dict)
        - model: Trained Keras model
        - best_f_w: Best weighted F-max score
        - history_dict: Training history dictionary
    """
    print(f"Loading data for aspect {aspect} with taxon_topk={config.taxon_topk}...")
    X_train, X_val, y_train, y_val, val_prot_ids, ontologies, gts, ns = load_data_for_aspect(
        aspect, taxon_topk=config.taxon_topk
    )
    print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # W&B setup
    run = None
    if use_wandb:
        import wandb
        from wandb.integration.keras import WandbMetricsLogger
        run = wandb.init(
            project=wandb_project,
            name=f"{aspect}_{config.name}",
            config={
                "aspect": aspect,
                **config.__dict__
            },
            reinit=True
        )
    
    # Build model
    model = build_model(config, X_train.shape[1], y_train.shape[1])
    
    # Setup callbacks
    cafa_callback = CafaEvalCallback(
        (X_val, y_val), val_prot_ids, ns, ontologies, gts[aspect]
    )
    
    callbacks = [
        cafa_callback,
        keras.callbacks.EarlyStopping(
            monitor="val_cafaeval_f_w",
            patience=early_stopping_patience,
            mode="max",
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_cafaeval_f_w",
            mode="max",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-5
        ),
    ]
    
    if use_wandb:
        from wandb.integration.keras import WandbMetricsLogger
        callbacks.append(WandbMetricsLogger())
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    best_f_w = cafa_callback.best_f_w
    print(f"\n>>> Best f_w: {best_f_w:.4f}")
    
    # Log and finish W&B
    if use_wandb and run is not None:
        import wandb
        wandb.log({"best_f_w": best_f_w})
        wandb.finish()
    
    # Save model
    if save_path:
        model.save(save_path)
        print(f"Model saved to: {save_path}")
    
    return model, best_f_w, history.history

