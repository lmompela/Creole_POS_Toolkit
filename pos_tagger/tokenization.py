"""
Tokenization and label alignment utilities for transformer-based POS tagging.
"""
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict, Optional


def tokenize_and_align_labels(
    sentences: List[Dict],
    tokenizer: PreTrainedTokenizer,
    label2id: Dict[str, int],
    max_length: int = 128
) -> Dict:
    """
    Tokenizes input sentences (split into words) and aligns word-level labels to subword tokens.

    Args:
      sentences: List of dicts, each with keys 'tokens' (List[str]) and 'tags' (List[str]).
      tokenizer: Hugging Face tokenizer with `is_split_into_words` support.
      label2id: Mapping from label string to integer ID.
      max_length: Maximum sequence length.

    Returns:
      A dict with tokenized inputs and 'labels' aligned to tokenizer output.
    """
    tokenized = tokenizer(
        [s['tokens'] for s in sentences],
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='np'
    )

    labels = []
    for i, sentence in enumerate(sentences):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label2id.get(sentence['tags'][word_idx], -100))
                prev_word_idx = word_idx
            else:
                label_ids.append(-100)
        labels.append(label_ids)

    tokenized['labels'] = labels
    return tokenized


def predict_transformer(
    tokens: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    device: Optional[torch.device] = None
) -> List[str]:
    """
    Predicts POS tags for a list of tokens and aligns them to words.

    Args:
      tokens: List of token strings (pre-tokenized words).
      tokenizer: Hugging Face tokenizer configured for the model.
      model: Hugging Face token-classification model.
      device: Torch device (if None, auto-detect CPU/CUDA).

    Returns:
      List of predicted label strings, one per input token.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize and get word_ids for alignment
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True,
        padding=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[0]  # (seq_len, num_labels)
    preds = torch.argmax(logits, dim=-1).cpu().tolist()

    word_ids = encoding.word_ids(batch_index=0)
    aligned = []
    prev_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_idx:
            aligned.append(preds[idx])
            prev_idx = word_idx

    labels = [model.config.id2label[p] for p in aligned]
    return labels
