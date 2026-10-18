---
title: Latin Scansion
emoji: 🏛️
colorFrom: yellow
colorTo: red
sdk: static
app_file: index.html
short_description: "TransScan: Latin scansion with Transformers, LSTM & CRF"
tags:
  - latin
  - poetry
  - scansion
  - sequence-labeling
  - transformer
  - lstm
  - crf
---

# Latin scansion with neural networks

Interactive demo for **TransScan: Scanning Latin dactylic verse using Transformers** by Yijun Wang, built on
[Ycreak/Latin_scansion_with_neural_networks](https://github.com/Ycreak/Latin_scansion_with_neural_networks) by Luuk Nolden
([thesis](https://theses.liacs.nl/pdf/2021-2022-NoldenL.pdf)).

Type lines of Latin verse and pick a model. Every syllable is labelled **long**, **short** or **elided**, and each line
is checked against dactylic hexameter and pentameter.

## Models and results

Everything is trained on `HEX_ELE-all` (56,500 Pedecerto hexameter and pentameter lines) with the repo's own training
code: `Latin_Transformer.create_model` / `Latin_LSTM.create_model` for 25 epochs (Adam, lr 0.001, validation split 0.2),
and `Latin_CRF.fit_model`. F1 per label:

| Model | Architecture | Long | Short | Elision |
|---|---|---|---|---|
| **Transformer Tiny** | 1 block, 4 heads, d=32, ffd=32 | 0.9787 | 0.9703 | 0.8191 |
| **Transformer Base** | 2 blocks, 4 heads, d=48, ffd=48 | 0.9837 | 0.9828 | 0.8794 |
| **Transformer Medium** | 3 blocks, 6 heads, d=48, ffd=48 | 0.9929 | 0.9908 | 0.9462 |
| **Transformer Large** | 4 blocks, 6 heads, d=64, ffd=64 | 0.9937 | 0.9926 | 0.9380 |
| BiLSTM (Nolden) | `lsnn/lstm.py` | 0.9902 | 0.9861 | 0.9946 |
| CRF (Nolden) | `lsnn/crf.py` | weighted 0.90 | | |

Scores are from TransScan Table 4 (the LSTM row cites Nolden) and, for the CRF, Nolden's thesis.

`transformer.py` builds a single `TransformerBlock`. To get the 2–4 block sizes, `build/train_transformers.py`
subclasses `Latin_Transformer` and overrides only `get_model`, which stacks the same block `num_blocks` times. The
original file is unchanged.

Transformer Large is listed with its paper results but isn't included in the demo.

## How it runs in the browser

Static Spaces have no Python server, so [Pyodide](https://pyodide.org) runs Python in the page. The repo's files are
shipped **unchanged**: `lsnn/crf.py`, `lsnn/lstm.py`, `lsnn/utilities.py`, `lsnn/config.py` and `transformer.py`. The
demo calls their own methods (`Latin_CRF.sentence_to_features`, `Latin_LSTM.predict_given_set`,
`Latin_Transformer.predict_given_set`).

TensorFlow, Keras and crfsuite aren't available in Pyodide, so `compat.py` does three things:
- stubs those imports;
- re-implements the trained networks' forward passes in NumPy;
- decodes the CRF's learned weights with Viterbi, as crfsuite does at prediction time.

`build/export.py` checks this against the real Keras and crfsuite models: on every syllable of `HEX_ELE-all` the
browser versions give **identical labels** to the trained models (100%).

Two demo-only adaptations:
- **Syllabification** (`syllabify.py`): typed text is split with the MQDQ syllabifier used to build the training data
  (vendored in `cltk_hax/`), following the corpus's *u/v* and *i/j* conventions.
- **Unseen syllables:** the original classes raise `KeyError` on a syllable that isn't in the corpus. Here it maps to
  the padding index and is shown in italics.

## Reproducing

From the root of the GitHub repo:

```bash
python hf_space/build/train_lstm.py   # original LSTM routine, 25 epochs, + calc_metrics_lstm-style scores
python hf_space/build/train_crf.py    # original CRF routine, exports model/crf.json
python hf_space/build/train_transformers.py tiny   # and base / medium / large (optional seed: large 2)
python hf_space/build/export.py       # exports weights + vocab, checks browser == Keras/crfsuite
```

## Limitations

The models only know dactylic verse. The syllabifier can't detect consonantal *i* inside words (*Lāvīnia*, *Iūlus*),
so those lines may come out a syllable too long. Lines over 29 tokens are cut from the front, as `pad_sequences` does
in the original code.
