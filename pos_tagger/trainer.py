"""
Training utilities for transformer-based POS tagging.
"""
import os
import logging
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

from pos_tagger.data_io import read_conllu_dataset
from pos_tagger.tokenization import tokenize_and_align_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants for learning rate scaling
_DEFAULT_BATCH_SIZE = 8
_BASE_LR = 5e-5


def scale_learning_rate(batch_size: int) -> float:
    """
    Dynamically scale the base learning rate according to batch size.
    """
    return _BASE_LR * (batch_size / _DEFAULT_BATCH_SIZE) ** 0.5


def get_user_lr(batch_size: int) -> float:
    """
    Suggest a learning rate and prompt the user to confirm or enter custom.
    """
    suggested = scale_learning_rate(batch_size)
    resp = input(f"Suggested learning rate for batch size {batch_size}: {suggested:.2e}. Accept? (Y/N) ").strip().lower()
    if resp == 'y':
        return suggested
    while True:
        try:
            custom = float(input("Enter a custom learning rate: "))
            return custom
        except ValueError:
            print("Invalid. Please enter a numeric learning rate.")


class SaveEveryNEpochsCallback(TrainerCallback):
    """
    Custom callback to save checkpoints every N epochs when using 'epoch' strategy.
    """
    def __init__(self, save_interval: int):
        self.save_interval = save_interval

    def on_epoch_end(self, args, state, control, **kwargs):
        current = int(state.epoch)
        # Only save on epochs 1, 1+interval, 1+2*interval, ...
        if (current - 1) % self.save_interval != 0:
            control.should_save = False
        return control


def compute_metrics(p):
    """
    Compute token-level accuracy (ignoring labels == -100).
    """
    logits, labels = p
    preds = np.argmax(logits, axis=2)
    total = 0
    correct = 0
    for pred_seq, label_seq in zip(preds, labels):
        for pr, lab in zip(pred_seq, label_seq):
            if lab == -100:
                continue
            total += 1
            if pr == lab:
                correct += 1
    acc = correct / total if total > 0 else 0.0
    return {"accuracy": torch.tensor(acc)}


def train_transformer(
    input_dir: str,
    model_output_dir: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    save_strategy: str = 'epoch',
    save_steps: int = 500,
    save_total_limit: int = None,
    save_epoch_interval: int = None
):
    """
    Train a transformer-based POS tagger.

    Args:
      input_dir: dir with 'train.conllu' and 'dev.conllu'
      model_output_dir: where to save model and tokenizer
      model_name: HF model identifier
      epochs: number of epochs
      batch_size: per-device batch size
      learning_rate: initial LR
      save_strategy: 'no', 'steps', or 'epoch'
      save_steps: steps interval for 'steps'
      save_total_limit: max checkpoints to keep
      save_epoch_interval: interval for 'epoch' strategy
    """
    # Load data
    train_data = read_conllu_dataset(os.path.join(input_dir, 'train.conllu'))
    dev_data = read_conllu_dataset(os.path.join(input_dir, 'dev.conllu'))

    # Build label mappings
    unique_labels = sorted({lbl for ex in train_data for lbl in ex['tags']})
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_tok = tokenize_and_align_labels(train_data, tokenizer, label2id)
    dev_tok = tokenize_and_align_labels(dev_data, tokenizer, label2id)

    import datasets
    train_ds = datasets.Dataset.from_dict(train_tok)
    dev_ds = datasets.Dataset.from_dict(dev_tok)

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # Prepare TrainingArguments
    args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_dir=os.path.join(model_output_dir, 'logs'),
        learning_rate=learning_rate
    )

    callbacks = []
    if save_strategy == 'epoch' and save_epoch_interval:
        callbacks.append(SaveEveryNEpochsCallback(save_epoch_interval))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    logging.info("Starting training...")
    trainer.train()

    # Save final model
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    logging.info(f"Model and tokenizer saved to {model_output_dir}")
