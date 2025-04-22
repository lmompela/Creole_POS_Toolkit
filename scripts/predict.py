#!/usr/bin/env python
"""
CLI for predicting POS tags with a trained transformer model.
Wraps pos_tagger.predictor.run_prediction with argument parsing.
"""
import os
import argparse
import logging

from pos_tagger.predictor import run_prediction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Predict POS tags using a transformer model and output evaluation metrics."
    )
    parser.add_argument("--input_conllu", required=True,
                        help="Path to input CoNLL-U file (test set)")
    parser.add_argument("--model_dir", required=True,
                        help="Directory of the trained transformer model")
    parser.add_argument("--output_conllu", default="predicted.conllu",
                        help="Path to write predicted CoNLL-U file")
    parser.add_argument("--eval_out", default="evaluation_report.txt",
                        help="Path to save simple evaluation report")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_conllu) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.eval_out) or '.', exist_ok=True)

    run_prediction(
        input_conllu=args.input_conllu,
        model_dir=args.model_dir,
        output_conllu=args.output_conllu,
        eval_out=args.eval_out
    )

if __name__ == '__main__':
    main()
