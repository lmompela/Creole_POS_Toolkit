"""
Prediction utilities for transformer-based POS tagging.
"""
import logging
from typing import List
from sklearn.metrics import accuracy_score, classification_report
import torch

from pos_tagger.data_io import read_conllu, write_conllu
from pos_tagger.tokenization import predict_transformer
from transformers import AutoTokenizer, AutoModelForTokenClassification

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_prediction(
    input_conllu: str,
    model_dir: str,
    output_conllu: str,
    eval_out: str
) -> None:
    """
    Load a transformer model and tokenizer, predict POS tags on an input CoNLL-U file,
    write out the predicted CoNLL-U and save simple evaluation metrics.
    """
    logging.info(f"Reading input file: {input_conllu}")
    sentences = read_conllu(input_conllu)

    logging.info(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    all_gold = []
    all_preds = []
    predicted = []

    for idx, sent in enumerate(sentences):
        # Collect tokens (skip comments and multiword tokens)
        tokens = []
        for entry in sent:
            if isinstance(entry, str):
                continue
            token_id, form = entry[0], entry[1]
            if "-" in token_id or "." in token_id:
                continue
            tokens.append(form)

        if tokens:
            preds = predict_transformer(tokens, tokenizer, model)
            # Replace UPOS tags in entries
            pred_idx = 0
            for j, entry in enumerate(sent):
                if isinstance(entry, list) and "-" not in entry[0] and "." not in entry[0]:
                    gold = entry[3]
                    pred = preds[pred_idx] if pred_idx < len(preds) else gold
                    all_gold.append(gold)
                    all_preds.append(pred)
                    entry[3] = pred
                    pred_idx += 1
        predicted.append(sent)

        if (idx + 1) % 50 == 0:
            logging.info(f"Processed {idx+1} sentences.")

    # Write predictions
    write_conllu(predicted, output_conllu)
    logging.info(f"Predicted CoNLL-U written to: {output_conllu}")

    # Evaluation
    acc = accuracy_score(all_gold, all_preds)
    report = classification_report(all_gold, all_preds, zero_division=0)
    eval_text = f"Overall Accuracy: {acc:.4f}\n\n{report}"
    print(eval_text)

    with open(eval_out, 'w', encoding='utf-8') as f:
        f.write(eval_text)
    logging.info(f"Evaluation report written to: {eval_out}")
