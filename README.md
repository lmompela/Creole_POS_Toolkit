# POS Tagger Toolkit

A modular, industry-standard Python package and CLI for training, predicting, and analyzing transformer-based POS tagging on CoNLL-U formatted data. Designed for low-resource languages (e.g., Creole) but extensible to any POS tagging task.

## Repository Structure

```text
pos_tagger/                 # Main Python package
  ├── data_io.py            # CoNLL-U read/write utilities
  ├── tokenization.py       # Subword-to-word alignment and tokenization helpers
  ├── trainer.py            # Training routines for transformer models
  ├── predictor.py          # Inference and simple evaluation routines
  └── analysis/             # Error analysis subpackage
      ├── error_analysis.py # All evaluation & diagnostic functions
      └── README.md         # Details for error analysis

scripts/                    # CLI entry-points
  ├── train.py              # Train transformer tagger
  ├── predict.py            # Predict and evaluate tags
  └── analyze.py            # Comprehensive error analysis

data/                       # (version-controlled) datasets
  ├── legacy/               # Legacy corpora (.gitignored)
  ├── flair/                # Flair-format data (.gitignored)
  ├── udpipe/               # UDPipe-format data (.gitignored)
  └── splits/               # train/dev/test splits

models/                     # Trained model checkpoints (.gitignored)
outputs/                    # Evaluation outputs (JSON, PNG, CSV)

README.md                   # You are here
requirements.txt            # Python dependencies
setup.py                    # Package installation script (optional)
.gitignore                  # Ignore logs, models, outputs, __pycache__, etc.
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lmompela/Martinican_Creole_POS_tagger.git
   cd pos_tagger
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. (Optional) Install as a package:
   ```bash
   pip install -e .
   ```
   This makes `pos_tagger` importable in Python and exposes CLI tools if configured in `setup.py`.

## CLI Usage

### 1. Train a POS Tagger

```bash
python scripts/train.py \
  --input_dir data/splits/mydataset \
  --model_dir models/transformer \
  --experiment_name creole_run \
  --transformer_model_name xlm-roberta-base \
  --transformer_epochs 5 \
  --transformer_batch_size 8 \
  --save_strategy epoch \
  --save_epoch_interval 2
```

- **`--input_dir`**: directory containing `train.conllu` and `dev.conllu`.  
- **`--model_dir`**: base path to save checkpoints.  
- **`--experiment_name`**: subfolder name under `model_dir`.  
- Other args configure model, epochs, batch size, and checkpoint strategy.

### 2. Predict and Evaluate

```bash
python scripts/predict.py \
  --input_conllu data/splits/mydataset/test.conllu \
  --model_dir models/transformer/creole_run \
  --output_conllu outputs/predicted.conllu \
  --eval_out outputs/evaluation_report.txt
```

- Writes predicted CoNLL-U file and simple accuracy/classification report.

### 3. Comprehensive Error Analysis

```bash
python scripts/analyze.py \
  data/splits/mydataset/test.conllu \
  outputs/predicted.conllu \
  data/splits/mydataset/train.conllu \
  --results_dir outputs/analysis \
  --target_words pou,ka \
  --ngram_n 2 --top_n 10 --min_freq 5
```

- Exports classification report, confusion matrix, LOESS plots, homonym & OOV error JSON, n-gram/collocation JSON, and KWIC CSV.

## Development & Contribution

- Follow modular design: add new analyses or data loaders in `pos_tagger/` submodules.  
- Update `requirements.txt` for new dependencies.  
- Create unit tests under a `tests/` directory and integrate with CI.  

## License

[MIT License](./LICENSE)

