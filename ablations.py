# ablations.py
"""
Systematic ablation study for CAFA-6 MLP models.
Run with: python ablations.py --aspect C
"""

import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
import wandb
from wandb.integration.keras import WandbMetricsLogger
import json
from datetime import datetime

# Import training utilities
from utils.training import (
    ModelConfig,
    build_model,
    load_data_for_aspect,
    CafaEvalCallback,
)

# ============== Ablation Configurations ==============

# Define baseline configuration
BASELINE = ModelConfig(name="baseline_2layer")

# Define ablation experiments - ONE change at a time from baseline
ABLATIONS = [
    # Baseline
    BASELINE,

    # --- Architecture depth ---
    ModelConfig(name="1layer_512", num_layers=1, layer_sizes=(512,)),
    ModelConfig(name="1layer_1024", num_layers=1, layer_sizes=(1024,)),
    ModelConfig(name="3layer", num_layers=3, layer_sizes=(2048, 1024, 512)),

    # --- Regularization type ---
    ModelConfig(name="bn_only", use_dropout=False, use_batchnorm=True),
    ModelConfig(name="dropout_only", use_dropout=True, use_batchnorm=False),
    ModelConfig(name="no_reg", use_dropout=False, use_batchnorm=False),

    # --- Dropout rates (without BN) ---
    ModelConfig(name="dropout_0.1", dropout=0.1, use_batchnorm=False),
    ModelConfig(name="dropout_0.2", dropout=0.2, use_batchnorm=False),
    ModelConfig(name="dropout_0.4", dropout=0.4, use_batchnorm=False),
    ModelConfig(name="dropout_0.5", dropout=0.5, use_batchnorm=False),

    # --- Learning rates ---
    ModelConfig(name="lr_5e-4", learning_rate=5e-4),
    ModelConfig(name="lr_1e-4", learning_rate=1e-4),
    ModelConfig(name="lr_5e-3", learning_rate=5e-3),

    # --- L2 regularization ---
    ModelConfig(name="l2_1e-4", l2_reg=1e-4),
    ModelConfig(name="l2_1e-3", l2_reg=1e-3),

    # --- Loss function ---
    ModelConfig(name="focal_loss", use_focal_loss=True),
    ModelConfig(name="focal_gamma_1", use_focal_loss=True, focal_gamma=1.0),

    # --- Batch size ---
    ModelConfig(name="batch_256", batch_size=256),
    ModelConfig(name="batch_1024", batch_size=1024),
]

# --- Taxon TopK ablation experiments ---
taxon_topk_ablation_points = [0, 5, 10, 20, 40, 80, 120, 160, 200]
for k in taxon_topk_ablation_points:
    ABLATIONS.append(
        ModelConfig(
            name=f"taxon_topk_{k}",
            num_layers=BASELINE.num_layers,
            layer_sizes=BASELINE.layer_sizes,
            learning_rate=BASELINE.learning_rate,
            dropout=BASELINE.dropout,
            use_batchnorm=BASELINE.use_batchnorm,
            use_dropout=BASELINE.use_dropout,
            batch_size=BASELINE.batch_size,
            epochs=BASELINE.epochs,
            use_focal_loss=BASELINE.use_focal_loss,
            focal_gamma=BASELINE.focal_gamma,
            l2_reg=BASELINE.l2_reg,
            taxon_topk=k
        )
    )

# --- Combination ablations ---
COMBINATION_ABLATIONS = [
    ModelConfig(name="combo_lr5e3_bn", 
                learning_rate=0.005, 
                use_batchnorm=True, 
                use_dropout=False),
    
    ModelConfig(name="combo_lr5e3_bn_b256", 
                learning_rate=0.005, 
                use_batchnorm=True, 
                use_dropout=False,
                batch_size=256),
    
    ModelConfig(name="combo_lr7e3_bn_b256", 
                learning_rate=0.007, 
                use_batchnorm=True, 
                use_dropout=False,
                batch_size=256),
    
    ModelConfig(name="combo_lr5e3_bn_b256_taxon200", 
                learning_rate=0.005, 
                use_batchnorm=True, 
                use_dropout=False,
                batch_size=256,
                taxon_topk=200),
]

# --- Final configuration for long training ---
FINAL_CONFIG = [
    ModelConfig(
        name="longer_reduce_lr_on_plateau",
        num_layers=2,
        layer_sizes=(1024, 512),
        learning_rate=0.005,
        use_batchnorm=True,
        use_dropout=True,
        batch_size=256,
        epochs=100,
    )
]


# ============== Ablation Runner ==============

def run_ablation(aspect: str = "C", ablations: list | None = None):
    """Run ablation experiments for a given aspect.
    
    Args:
        aspect: GO aspect ("C", "F", or "P")
        ablations: List of ModelConfig objects to run. Defaults to FINAL_CONFIG.
    
    Returns:
        List of result dictionaries sorted by best f_w score
    """
    if ablations is None:
        ablations = FINAL_CONFIG

    results = []

    for config in ablations:
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"{'='*60}")

        print(f"Loading data for aspect {aspect} with taxon_topk={config.taxon_topk}...")
        X_train, X_val, y_train, y_val, val_prot_ids, ontologies, gts, ns = load_data_for_aspect(
            aspect, taxon_topk=config.taxon_topk
        )
        print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        
        # Initialize W&B
        run = wandb.init(
            project="cafa6-ablation",
            name=f"{aspect}_{config.name}",
            config={
                "aspect": aspect,
                "experiment": config.name,
                **config.__dict__
            },
            reinit=True
        )

        # Build model
        model = build_model(config, X_train.shape[1], y_train.shape[1])

        # Callbacks
        cafa_callback = CafaEvalCallback(
            (X_val, y_val), val_prot_ids, ns, ontologies, gts[aspect]
        )

        callbacks = [
            cafa_callback,
            WandbMetricsLogger(),
            keras.callbacks.EarlyStopping(
                monitor="val_cafaeval_f_w",
                patience=10,
                mode="max",
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_cafaeval_f_w",
                mode="max",
                factor=0.5,
                patience=8,
                min_lr=1e-5
            ),
        ]

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Record results
        best_f_w = cafa_callback.best_f_w
        result = {
            "name": config.name,
            "best_f_w": best_f_w,
            "config": config.__dict__,
            "epochs_trained": len(history.history['loss'])
        }
        results.append(result)

        print(f"\n>>> Best f_w for {config.name}: {best_f_w:.4f}")

        wandb.log({"best_f_w": best_f_w})
        wandb.finish()

        # Clear memory
        del model
        keras.backend.clear_session()

    # Save results
    results_sorted = sorted(results, key=lambda x: x['best_f_w'], reverse=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ablation_results_{aspect}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results_sorted, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION RESULTS (sorted by best f_w):")
    print(f"{'='*60}")
    for r in results_sorted:
        print(f"{r['name']:25s} | f_w: {r['best_f_w']:.4f}")

    print(f"\nResults saved to: {results_file}")

    return results_sorted


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CAFA-6 ablation studies")
    parser.add_argument("--aspect", type=str, default="C", choices=["C", "F", "P"],
                       help="GO aspect to train on")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only 5 key experiments")
    parser.add_argument("--taxon_topk_ablation", action="store_true", 
                       help="Run ONLY the taxon_topk ablation set")
    parser.add_argument("--combinations", action="store_true",
                       help="Run combination ablations")
    args = parser.parse_args()
    
    if args.quick:
        # Quick ablation - just the most important comparisons
        quick_ablations = [
            ModelConfig(name="baseline_2layer"),
            ModelConfig(name="1layer_1024", num_layers=1, layer_sizes=(1024,)),
            ModelConfig(name="dropout_only_0.3", use_dropout=True, use_batchnorm=False),
            ModelConfig(name="focal_loss", use_focal_loss=True),
            ModelConfig(name="lr_5e-4", learning_rate=5e-4),
        ]
        run_ablation(args.aspect, quick_ablations)
    elif args.taxon_topk_ablation:
        taxon_topk_ablations = [c for c in ABLATIONS if c.name.startswith("taxon_topk_")]
        run_ablation(args.aspect, taxon_topk_ablations)
    elif args.combinations:
        run_ablation(args.aspect, COMBINATION_ABLATIONS)
    else:
        run_ablation(args.aspect)
