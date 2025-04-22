#!/usr/bin/env python
"""
CLI for comprehensive error analysis on POS tagging outputs.
Runs classification report, confusion matrix, LOESS F1 vs. support,
homonym error rates, OOV error rates, and optional n-gram/collocation/KWIC analyses.
"""
import os
import argparse
import logging
import json

from pos_tagger.data_io import read_conllu, read_conllu_sentences, read_conllu_dataset
from pos_tagger.analysis.error_analysis import (
    export_classification_report,
    plot_confusion_matrix,
    analyze_f1_vs_support,
    compute_homonym_errors,
    compute_oov_error_rate,
    extract_texts,
    targeted_ngram_analysis,
    targeted_collocation_analysis,
    extract_kwic_for_targets,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(
        description="Run full POS tagging error analyses and export results."
    )
    parser.add_argument("gold_file", help="Gold-standard CoNLL-U file")
    parser.add_argument("pred_file", help="Predicted CoNLL-U file")
    parser.add_argument("train_file", nargs="?", default=None,
                        help="Optional: training CoNLL-U file for OOV analysis")
    parser.add_argument("--results_dir", default="results",
                        help="Directory to save all analysis outputs")
    parser.add_argument("--target_words", default="",
                        help="Comma-separated tokens for n-gram/collocation/KWIC analysis")
    parser.add_argument("--ngram_n", type=int, default=2,
                        help="n for n-gram analysis (default 2)")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Top N items for n-gram/collocation")
    parser.add_argument("--min_freq", type=int, default=3,
                        help="Min frequency for collocation filter")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load sentences
    logging.info("Loading gold and predicted sentences...")
    gold_sents = read_conllu_sentences(args.gold_file)
    pred_sents = read_conllu_sentences(args.pred_file)

    # Flatten tags
    gold_tags = [tok[3] for sent in gold_sents for tok in sent]
    pred_tags = [tok[3] for sent in pred_sents for tok in sent]

    # Classification report
    logging.info("Exporting classification report...")
    cls_report = export_classification_report(
        gold_tags, pred_tags, args.results_dir)

    # Confusion matrix
    logging.info("Plotting confusion matrix...")
    plot_confusion_matrix(gold_tags, pred_tags, args.results_dir)

    # LOESS analysis
    logging.info("Performing LOESS F1 vs. support analysis...")
    analyze_f1_vs_support(gold_tags, pred_tags, results_dir=args.results_dir)

    # Homonym errors
    logging.info("Computing homonym error rates...")
    hom_errors = compute_homonym_errors(gold_sents, pred_sents)
    hom_path = os.path.join(args.results_dir, 'homonym_errors.json')
    with open(hom_path, 'w', encoding='utf-8') as f:
        json.dump(hom_errors, f, ensure_ascii=False, indent=2)
    logging.info(f"Homonym error rates saved to: {hom_path}")

    # OOV errors
    if args.train_file:
        logging.info("Computing OOV error rates...")
        train_sents = read_conllu_sentences(args.train_file)
        oov_rate, oov_err, oov_total, oov_details = compute_oov_error_rate(
            gold_sents, pred_sents, train_sents)
        oov_path = os.path.join(args.results_dir, 'oov_errors.json')
        with open(oov_path, 'w', encoding='utf-8') as f:
            json.dump({
                'error_rate': oov_rate,
                'error_count': oov_err,
                'total_oov': oov_total,
                'details': oov_details
            }, f, ensure_ascii=False, indent=2)
        logging.info(f"OOV error results saved to: {oov_path}")

    # Targeted lexical analyses
    if args.target_words:
        targets = [t.strip() for t in args.target_words.split(',') if t.strip()]
        texts = extract_texts(gold_sents)

        # n-grams
        logging.info("Running targeted n-gram analysis...")
        ngram_res = {t: targeted_ngram_analysis(texts, t, n=args.ngram_n, top_n=args.top_n)
                     for t in targets}
        with open(os.path.join(args.results_dir, 'ngram_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(ngram_res, f, ensure_ascii=False, indent=2)

        # collocations
        logging.info("Running targeted collocation analysis...")
        colloc_res = {t: targeted_collocation_analysis(texts, t, min_freq=args.min_freq, top_n=args.top_n)
                      for t in targets}
        with open(os.path.join(args.results_dir, 'collocation_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(colloc_res, f, ensure_ascii=False, indent=2)

        # KWIC
        logging.info("Extracting KWIC contexts...")
        kwic_df = extract_kwic_for_targets(texts, targets)
        kwic_csv = os.path.join(args.results_dir, 'kwic_extracted.csv')
        kwic_df.to_csv(kwic_csv, index=False)
        logging.info(f"KWIC entries saved to: {kwic_csv}")

    logging.info("Analysis complete.")

if __name__ == '__main__':
    main()
