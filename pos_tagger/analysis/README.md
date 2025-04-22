# Error Analysis Module

This folder contains utilities and scripts for in-depth error analysis of POS tagging outputs. It centralizes functions to evaluate, visualize, and export various diagnostics, including classification reports, confusion matrices, LOESS-based performance trends, and specialized analyses for homonymous and OOV tokens.

## Contents

- **error_analysis.py**: Core module with functions for:
  - **Classification Report**: `export_classification_report(gold_tags, pred_tags, results_dir)`
  - **Confusion Matrix**: `plot_confusion_matrix(gold_tags, pred_tags, results_dir)`
  - **LOESS F1 vs. Support**: `analyze_f1_vs_support(gold_tags, pred_tags, target_f1, results_dir)`
  - **Homonym Error Rates**:
    - `get_homonymous_tokens(sentences)`
    - `compute_homonym_errors(gold_sents, pred_sents)`
  - **OOV Error Rates**: `compute_oov_error_rate(gold_sents, pred_sents, train_sents)`
  - **Targeted Lexical Analyses**:
    - `targeted_ngram_analysis(texts, target, n, top_n)`
    - `targeted_collocation_analysis(texts, target, min_freq, top_n)`
    - `extract_kwic_for_targets(texts, targets)`
  - **Helper Functions**:
    - `extract_texts(sentences)`
    - `custom_tokenize(text)` / `tokenize_lower(text)`
    - `extract_kwic(text, target)`

- **scripts/analyze.py**: CLI entry point to run all analyses in batch. Supports:
  - Exporting classification report and confusion matrix images.
  - LOESS smoothing of F1 vs. support and threshold detection.
  - Homonym and OOV error rate computation (outputs JSON).
  - Optional targeted n-gram, collocation, and KWIC analyses (JSON/CSV).

## Usage Example

```bash
python scripts/analyze.py \
  path/to/gold.conllu path/to/predicted.conllu \
  --results_dir path/to/results \
  --target_words pou,ka \
  --ngram_n 2 --top_n 10 --min_freq 3
```

If no need for KWIC just use: 

```bash
python scripts/analyze.py \
  path/to/gold.conllu path/to/predicted.conllu \
  --results_dir path/to/results
```

For OOV Error Rate, use:

```bash
python scripts/analyze.py \
  path/to/gold.conllu path/to/predicted.conllu path/to/train.conllu \
  --results_dir path/to/results
```

After running, the `results` directory will contain:

- `classification_report.txt`
- `confusion_matrix.png` and `confusion_matrix.txt`
- `f1_vs_support.png` and `loess_results.txt`
- `homonym_errors.json`
- `oov_errors.json` (if `train.conllu` provided)
- `ngram_analysis.json`, `collocation_analysis.json`
- `kwic_extracted.csv`

## Dependencies

- Python 3.7+
- `numpy`, `pandas`, `matplotlib`, `statsmodels`
- `scikit-learn`, `nltk`
- `transformers` (for type consistency, though not used directly)

Ensure you have run:

```bash
pip install -r requirements.txt
```

