"""
Error analysis utilities for POS tagging.
Includes classification report export, confusion matrix plotting,
F1 vs. support (LOESS) analysis, homonym error rates,
OOV error rates, and targeted n-gram, collocation, and KWIC analyses.
"""
import os
import re
import logging
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

from pos_tagger.data_io import read_conllu, read_conllu_sentences

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Classification Report ---

def export_classification_report(gold_tags, pred_tags, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    report_str = classification_report(gold_tags, pred_tags, zero_division=0)
    out_path = os.path.join(results_dir, 'classification_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_str)
    logging.info(f"Classification report saved to {out_path}")
    return report_str

# --- Confusion Matrix ---

def plot_confusion_matrix(gold_tags, pred_tags, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    labels = sorted(set(gold_tags) | set(pred_tags))
    cm = confusion_matrix(gold_tags, pred_tags, labels=labels)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix for POS Tagging")
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    cm_png = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_png, dpi=300)
    plt.close()
    cm_txt = os.path.join(results_dir, 'confusion_matrix.txt')
    np.savetxt(cm_txt, cm, fmt='%d', delimiter='\t', header='\t'.join(labels), comments='')
    logging.info(f"Confusion matrix files saved: {cm_png}, {cm_txt}")
    return cm

# --- F1 vs. Support (LOESS) ---

def analyze_f1_vs_support(gold_tags, pred_tags, target_f1=0.80, results_dir='.'):  
    os.makedirs(results_dir, exist_ok=True)
    labels = sorted(set(gold_tags) | set(pred_tags))
    report = classification_report(gold_tags, pred_tags, labels=labels, output_dict=True, zero_division=0)
    supports = [gold_tags.count(lbl) for lbl in labels]
    f1s = [report[lbl]['f1-score'] for lbl in labels]
    loess = sm.nonparametric.lowess(f1s, supports, frac=0.3)
    threshold = next((sup for sup, f1 in loess if f1 >= target_f1), None)
    plt.figure(figsize=(10,6))
    plt.scatter(supports, f1s, alpha=0.6)
    plt.plot(loess[:,0], loess[:,1], '--')
    if threshold is not None:
        plt.axvline(threshold, linestyle=':')
    plt.xlabel('Support (Frequency)')
    plt.ylabel('F1 Score')
    plt.title('LOESS: F1 vs. Support')
    out_png = os.path.join(results_dir, 'f1_vs_support.png')
    plt.savefig(out_png, dpi=300)
    plt.close()
    loess_txt = os.path.join(results_dir, 'loess_results.txt')
    with open(loess_txt, 'w') as f:
        f.write('Support\tLOESS_F1\n')
        for sup, f1 in loess:
            f.write(f"{sup:.0f}\t{f1:.4f}\n")
        if threshold is not None:
            f.write(f"Threshold for F1>={target_f1}: {threshold:.0f}\n")
    logging.info(f"LOESS analysis saved: {out_png}, {loess_txt}")
    return loess, threshold

# --- Homonym Error Rates ---

def get_homonymous_tokens(sentences):
    token_map = defaultdict(set)
    for sent in sentences:
        for token in sent:
            if isinstance(token, list):
                tid, form, _, upos = token[0], token[1].lower(), token[2], token[3]
                if '-' in tid or '.' in tid:
                    continue
                token_map[form].add(upos)
    return {w: tags for w, tags in token_map.items() if len(tags) > 1}


def compute_homonym_errors(gold_sents, pred_sents):
    homonyms = get_homonymous_tokens(gold_sents)
    stats = {w: [0,0] for w in homonyms}
    details = defaultdict(Counter)
    for g, p in zip(gold_sents, pred_sents):
        if len(g) != len(p): continue
        for gt, pt in zip(g, p):
            if isinstance(gt, list) and '-' not in gt[0] and '.' not in gt[0]:
                word = gt[1].lower()
                if word in homonyms:
                    stats[word][1] += 1
                    if gt[3] != pt[3]:
                        stats[word][0] += 1
                        details[word][(gt[3], pt[3])] += 1
    error_rates = {}
    for w, (err, tot) in stats.items():
        if tot > 0:
            error_rates[w] = (err/tot, err, tot, list(details[w].items()))
    return error_rates

# --- OOV Error Rates ---

def compute_oov_error_rate(gold_sents, pred_sents, train_sents):
    """
    Computes error rate for out-of-vocabulary tokens.

    Args:
      gold_sents: List of gold CoNLL-U sentences
      pred_sents: List of predicted CoNLL-U sentences
      train_sents: List of gold CoNLL-U sentences from training set

    Returns:
      Tuple: (error_rate, error_count, total_oov, details dict)
    """
    # Build training vocab
    vocab = set()
    for sent in train_sents:
        for token in sent:
            if isinstance(token, list) and '-' not in token[0] and '.' not in token[0]:
                vocab.add(token[1].lower())
    # Compute OOV errors
    error_count = 0
    total_oov = 0
    details = defaultdict(Counter)
    for g, p in zip(gold_sents, pred_sents):
        if len(g) != len(p):
            continue
        for gt, pt in zip(g, p):
            if isinstance(gt, list) and '-' not in gt[0] and '.' not in gt[0]:
                word = gt[1].lower()
                if word not in vocab:
                    total_oov += 1
                    if gt[3] != pt[3]:
                        error_count += 1
                        details[word][(gt[3], pt[3])] += 1
    rate = error_count / total_oov if total_oov > 0 else 0.0
    return rate, error_count, total_oov, dict(details)

# --- KWIC, N-gram, Collocations ---

def extract_texts(sentences, form_index=1):
    texts = []
    for sent in sentences:
        words = [t[form_index] for t in sent if isinstance(t, list) and '-' not in t[0] and '.' not in t[0]]
        texts.append(' '.join(words))
    return texts


def custom_tokenize(text):
    return RegexpTokenizer(r"[A-Za-zÀ-ÖØ-öø-ÿ]+").tokenize(text)


def tokenize_lower(text):
    return [t.lower() for t in custom_tokenize(text)]


def targeted_ngram_analysis(texts, target, n=2, top_n=10):
    ctr = Counter()
    for t in texts:
        toks = tokenize_lower(t)
        for gram in ngrams(toks, n):
            if target.lower() in gram:
                ctr[gram] += 1
    return ctr.most_common(top_n)


def targeted_collocation_analysis(texts, target, min_freq=3, top_n=10):
    words = []
    for t in texts:
        words.extend(tokenize_lower(t))
    finder = BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(min_freq)
    measures = BigramAssocMeasures()
    collocs = []
    for bg, freq in finder.ngram_fd.items():
        if target.lower() in bg:
            score = finder.score_ngram(measures.pmi, bg[0], bg[1])
            collocs.append((bg, score, freq))
    return sorted(collocs, key=lambda x: x[1], reverse=True)[:top_n]


def extract_kwic(text, target):
    pat = re.compile(r"(.*?)(\b" + re.escape(target) + r"\b)(.*)", flags=re.IGNORECASE | re.DOTALL)
    m = pat.search(text)
    return (m.group(1).strip(), m.group(2), m.group(3).strip()) if m else None


def extract_kwic_for_targets(texts, targets):
    entries = []
    for t in texts:
        for tgt in targets:
            k = extract_kwic(t, tgt)
            if k:
                entries.append({'target': tgt, 'before': k[0], 'after': k[2]})
    return pd.DataFrame(entries)
