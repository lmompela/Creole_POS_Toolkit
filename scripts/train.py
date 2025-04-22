#!/usr/bin/env python
"""
CLI for training transformer-based POS tagger.
Wraps pos_tagger.trainer.train_transformer with argument parsing.
"""
import os
import argparse
import logging

from pos_tagger.trainer import (
    train_transformer,
    get_user_lr
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(
        description="Train a transformer-based POS tagger on CoNLL-U splits."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing train.conllu and dev.conllu splits")
    parser.add_argument("--model_dir", default="models/transformer",
                        help="Base directory to save model checkpoints")
    parser.add_argument("--experiment_name", default=None,
                        help="(Optional) subfolder under model_dir to name this run")
    parser.add_argument("--transformer_model_name", default="xlm-roberta-base",
                        help="HuggingFace transformer model identifier")
    parser.add_argument("--transformer_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--transformer_batch_size", type=int, default=8,
                        help="Batch size per device")
    parser.add_argument("--save_strategy", choices=["no","epoch","steps"], default="epoch",
                        help="Checkpoint saving strategy")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save every N steps when using 'steps' strategy")
    parser.add_argument("--save_total_limit", type=int, default=None,
                        help="Max number of checkpoints to keep")
    parser.add_argument("--save_epoch_interval", type=int, default=None,
                        help="When using 'epoch' strategy, save every N epochs")
    args = parser.parse_args()

    # Determine output path
    if args.experiment_name:
        output_dir = os.path.join(args.model_dir, args.experiment_name)
    else:
        output_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)

    # Suggest or confirm learning rate
    lr = get_user_lr(args.transformer_batch_size)

    # Call training
    train_transformer(
        input_dir=args.input_dir,
        model_output_dir=output_dir,
        model_name=args.transformer_model_name,
        epochs=args.transformer_epochs,
        batch_size=args.transformer_batch_size,
        learning_rate=lr,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_epoch_interval=args.save_epoch_interval
    )

if __name__ == '__main__':
    main()
