# CLI Scripts

This folder contains command-line entry points to train, predict, and analyze your POS tagging models using the `pos_tagger` package.

## Contents

- **train.py**  
  Train a transformer-based POS tagger on pre-split CoNLL-U data.

- **predict.py**  
  Generate POS tag predictions for a test set and compute a simple evaluation report.

- **analyze.py**  
  Run comprehensive error analyses (classification report, confusion matrix, LOESS F1 vs. support, homonym and OOV error rates, and targeted lexical analyses).

## Usage Overview

All scripts assume you have installed dependencies (e.g., via `pip install -r requirements.txt`) and have the `pos_tagger` package available.

### 1. `train.py`
```bash
python scripts/train.py \
  --input_dir <path/to/data/splits> \
  --model_dir <path/to/save/models> \
  [--experiment_name <name>] \
  [--transformer_model_name <hf-model>] \
  [--transformer_epochs <N>] \
  [--transformer_batch_size <B>] \
  [--save_strategy <no|epoch|steps>] \
  [--save_steps <N>] \
  [--save_total_limit <N>] \
  [--save_epoch_interval <N>]
```
Key arguments:
- `--input_dir`: directory containing `train.conllu` and `dev.conllu` splits.  
- `--model_dir`: base directory for saving model checkpoints.  
- `--experiment_name`: optional subfolder name under `model_dir`.  
- Checkpoint options: `--save_strategy`, `--save_steps`, `--save_total_limit`, `--save_epoch_interval`.

### 2. `predict.py`
```bash
python scripts/predict.py \
  --input_conllu <path/to/test.conllu> \
  --model_dir <path/to/saved/model> \
  [--output_conllu <path/to/predictions.conllu>] \
  [--eval_out <path/to/evaluation_report.txt>]
```
- Reads gold test file and saved model.  
- Writes predicted CONLLU file and a text evaluation report.

### 3. `analyze.py`
```bash
python scripts/analyze.py \
  <gold.conllu> <predicted.conllu> [<train.conllu>] \
  [--results_dir <path/to/results>] \
  [--target_words <tok1,tok2,...>] \
  [--ngram_n <N>] [--top_n <N>] [--min_freq <N>]
```
- **Required**: paths to gold and predicted CoNLL-U files.  
- **Optional**: training split for OOV analysis.  
- **`--results_dir`**: where to save all outputs (\*.txt, \*.png, \*.json, \*.csv).  
- **Targeted analyses**: specify `--target_words` to run n-gram, collocation, and KWIC.

## Output

- **train.py**: model checkpoints and logs in `--model_dir`.
- **predict.py**: `output_conllu`, `eval_out` files.
- **analyze.py**: in `--results_dir`, you’ll find:
  - `classification_report.txt`  
  - `confusion_matrix.png` & `.txt`  
  - `f1_vs_support.png` & `loess_results.txt`  
  - `homonym_errors.json`  
  - `oov_errors.json` (if training file provided)  
  - `ngram_analysis.json`, `collocation_analysis.json`, and `kwic_extracted.csv` (if targets)


---

For any issues or contributions, feel free to open a GitHub issue or pull request.

