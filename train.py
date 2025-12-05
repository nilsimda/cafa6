#!/usr/bin/env python
"""
Standalone training script for CAFA-6 MLP models.

Examples:
    # Train with default config
    python train.py --aspect C
    
    # Train with custom parameters
    python train.py --aspect F --lr 0.005 --epochs 50 --batch-size 256
    
    # Train with W&B logging
    python train.py --aspect P --wandb --wandb-project my-project
    
    # Train and save the model
    python train.py --aspect C --save keras_models/my_model.keras
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
from utils.training import ModelConfig, train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CAFA-6 MLP model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument("--aspect", type=str, default="C", choices=["C", "F", "P"],
                       help="GO aspect to train on")
    
    # Architecture
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of hidden layers")
    parser.add_argument("--layer-sizes", type=int, nargs="+", default=[1024, 512],
                       help="Units per layer")
    
    # Training
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Maximum epochs")
    
    # Regularization
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--no-dropout", action="store_true",
                       help="Disable dropout")
    parser.add_argument("--no-batchnorm", action="store_true",
                       help="Disable batch normalization")
    parser.add_argument("--l2-reg", type=float, default=0.0,
                       help="L2 regularization strength")
    
    # Loss
    parser.add_argument("--focal-loss", action="store_true",
                       help="Use focal loss instead of BCE")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Focal loss gamma parameter")
    
    # Features
    parser.add_argument("--taxon-topk", type=int, default=77,
                       help="Top-k taxon IDs for one-hot encoding (0 = no taxon features)")
    
    # Callbacks
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--reduce-lr-patience", type=int, default=8,
                       help="Reduce LR on plateau patience")
    
    # Logging & Saving
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="cafa6",
                       help="W&B project name")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save trained model")
    parser.add_argument("--name", type=str, default="custom",
                       help="Name for this training run")
    
    # GPU
    parser.add_argument("--gpu", type=str, default=None,
                       help="GPU device ID (e.g., '0' or '1')")
    
    parser.add_argument("-v", "--verbose", type=int, default=1,
                       help="Verbosity level (0, 1, or 2)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Build config from args
    config = ModelConfig(
        name=args.name,
        num_layers=args.num_layers,
        layer_sizes=tuple(args.layer_sizes),
        learning_rate=args.lr,
        dropout=args.dropout,
        use_batchnorm=not args.no_batchnorm,
        use_dropout=not args.no_dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        l2_reg=args.l2_reg,
        taxon_topk=args.taxon_topk,
    )
    
    print(f"Training configuration:")
    print(f"  Aspect: {args.aspect}")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print()
    
    # Train
    model, best_f_w, history = train_model(
        config=config,
        aspect=args.aspect,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        save_path=args.save,
        verbose=args.verbose,
    )
    
    print(f"\nTraining complete!")
    print(f"  Best F-max (weighted): {best_f_w:.4f}")
    print(f"  Epochs trained: {len(history['loss'])}")


if __name__ == "__main__":
    main()

