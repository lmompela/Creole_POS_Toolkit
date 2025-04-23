# Martinican Creole POS Tagger Toolkit

A modular Python package and CLI for training, predicting, and analyzing transformer-based POS taggers on CoNLL‑U data. Includes three ready‑to‑use, fine‑tuned models for Martinican Creole:

| Model Repository                                              | Base Model       | Accuracy |
| ------------------------------------------------------------- | ---------------- | -------- |
| `lmompelat/xlm-r-base-martinican-pos-tagger`                  | XLM‑RoBERTa       | 85%      |
| `lmompelat/mbert-martinican-pos-tagger`                       | mBERT             | 91%      |
| `lmompelat/creoleval-martinican-pos-tagger`                   | XLM‑RoBERTa + finetune | 92% |

## Installation

Clone and install dependencies:
```bash
git clone https://github.com/lmompela/Martinican_Creole_POS_tagger.git
cd Martinican_Creole_POS_tagger
python3 -m venv .venv           # or your preferred venv tool
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional: editable install
pip install -e .[torch]
```

## Pre-trained Models

You can load any of the listed models directly from the Hugging Face Hub:

```python
from transformers import pipeline

taggers = {
    "xlm-r": "lmompelat/xlm-r-base-martinican-pos-tagger",
    "mbert": "lmompelat/mbert-martinican-pos-tagger",
    "creoleval": "lmompelat/creoleval-martinican-pos-tagger",
}

tagger = pipeline(
    "token-classification", 
    model=taggers["creoleval"], 
    tokenizer=taggers["creoleval"]
)

text = "Mwen ka alé an lékol pou estudiar Kréyol."
predictions = tagger(text)
print(predictions)
```

## CLI Usage

### 1. Training

```bash
python scripts/train.py \
  --input_dir data/splits/mydataset \
  --model_dir models/transformer \
  --experiment_name my_experiment \
  --transformer_model_name xlm-roberta-base \
  --transformer_epochs 5 \
  --transformer_batch_size 8
```

### 2. Prediction

```bash
python scripts/predict.py \
  --input_conllu data/splits/mydataset/test.conllu \
  --model_dir lmompelat/creoleval-martinican-pos-tagger \
  --output_conllu outputs/predicted.conllu \
  --eval_out outputs/evaluation_report.txt
```

### 3. Error Analysis

```bash
python scripts/analyze.py \
  data/splits/mydataset/test.conllu \
  outputs/predicted.conllu \
  data/splits/mydataset/train.conllu \
  --results_dir outputs/analysis \
  --target_words pou,ka \
  --ngram_n 2 --top_n 10 --min_freq 5
```

## Project Structure (abbreviated)

```
pos_tagger/    # core modules
scripts/       # CLI entry points
models/        # local checkpoint folder (ignored by git)
data/          # version-controlled datasets
outputs/       # evaluation outputs
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

