"""
Data I/O utilities for CoNLL-U processing and general file operations.
"""
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_conllu(filepath):
    """
    Reads a CoNLL-U file, preserving comment lines.
    Returns:
      - List of sentences, each a list of entries:
        * comment lines as strings (e.g., "# text = ...")
        * token lines as lists of columns (fields)
    """
    sentences = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            # Sentence boundary
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            # Preserve comments
            if line.startswith('#'):
                current.append(line)
            else:
                parts = line.split('\t')
                if len(parts) >= 4:
                    current.append(parts)
        # Add last sentence if file doesn't end with newline
        if current:
            sentences.append(current)
    logging.info(f"Loaded {len(sentences)} sentences from {filepath} (including comments)")
    return sentences


def read_conllu_sentences(filepath):
    """
    Reads a CoNLL-U file, ignoring comment lines.
    Returns:
      - List of sentences, each a list of token columns (lists of fields).
    """
    raw = read_conllu(filepath)
    sentences = []
    for sent in raw:
        tokens = [entry for entry in sent if not isinstance(entry, str)]
        if tokens:
            sentences.append(tokens)
    logging.info(f"Parsed {len(sentences)} token-only sentences from {filepath}")
    return sentences


def read_conllu_dataset(filepath):
    """
    Reads a CoNLL-U file formatted for training (pre-tokenized).
    Each sentence block is separated by a blank line.

    Returns:
      - List of dicts, each with:
         'tokens': List[str],
         'tags': List[str],
         'raw_text': str (joined tokens)
    """
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read().strip().split('\n\n')
        for block in raw:
            tokens, tags = [], []
            for line in block.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 4:
                    tokens.append(parts[1])
                    tags.append(parts[3])
            if tokens:
                sentences.append({
                    'tokens': tokens,
                    'tags': tags,
                    'raw_text': ' '.join(tokens)
                })
    logging.info(f"Loaded {len(sentences)} training examples from {filepath}")
    return sentences


def write_conllu(sentences, output_file):
    """
    Writes sentences to a CoNLL-U file.

    Args:
      sentences: List of sentences, each a list of entries:
        - comment lines as strings
        - token lines as lists of fields (updated tags included)
      output_file: Path to write the .conllu output.
    """
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in sentences:
            for entry in sent:
                if isinstance(entry, str):
                    f.write(entry + '\n')
                else:
                    f.write('\t'.join(entry) + '\n')
            f.write('\n')
    logging.info(f"Wrote {len(sentences)} sentences to {output_file}")
