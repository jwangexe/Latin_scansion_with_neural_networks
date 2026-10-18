"""Runs in the browser under Pyodide. Scans verse with the ORIGINAL repo classes
(lsnn/crf.py, lsnn/lstm.py, transformer.py), backed by compat.py, and renders HTML."""
import html
import json
import re

import compat

compat.install_stubs()

from lsnn.crf import Latin_CRF  # noqa: E402  (imports need the stubs above)
from lsnn.lstm import Latin_LSTM  # noqa: E402
from transformer import Latin_Transformer  # noqa: E402

from syllabify import SPACE, line_to_words, words_to_sequence  # noqa: E402


class _Vocab(dict):
    """word2idx exactly as the original classes build it. Typed text can contain
    syllables that never occur in the corpus (the original raises KeyError); those
    map to the padding index and are flagged as unseen in the UI."""

    def __missing__(self, key):
        return self["padding"]


def _neural(cls, model):
    """Original class with the state its __init__ derives from the pickles (built
    at export time, since the 12 MB corpus isn't shipped to the browser)."""
    obj = object.__new__(cls)
    obj.anceps_label = False
    obj.word2idx = _Vocab({s: i for i, s in enumerate(VOCAB["unique_syllables"])})
    obj.label2idx = {t: i for i, t in enumerate(cls.LABELS)}
    obj.max_sentence_length = VOCAB["max_sentence_length"]
    obj.model = model
    return obj


VOCAB = json.load(open("model/vocab.json"))
MODELS = {
    # TransScan Transformer sizes (paper, Table 3), built with transformer.py
    **{f"transformer_{size}": _neural(Latin_Transformer, compat.NumpyTransformer(f"model/transformer_{size}_weights.npz"))
       for size in ("tiny", "base", "medium")},  # Large: see README (did not reproduce)
    "lstm": _neural(Latin_LSTM, compat.NumpyLSTM("model/lstm_weights.npz")),
}
CRF = Latin_CRF()
CRF_TAGGER = compat.CRFTagger("model/crf.json")

MARK = {"long": "—", "short": "∪", "elision": "×"}

# Feet as L/S strings; the final syllable of a line is anceps.
HEXAMETER = re.compile(r"^((?:LSS|LL){5})(L[LS])$")
PENTAMETER = re.compile(r"^(LSS|LL)(LSS|LL)(L)(LSS)(LSS)([LS])$")


def _fmt(feet):
    return " | ".join(" ".join(MARK["long" if c == "L" else "short"] for c in f) for f in feet)


def fit_meter(pattern: str) -> tuple[str | None, str]:
    """Match an L/S pattern (elisions removed) against hexameter / pentameter."""
    if HEXAMETER.match(pattern):
        return "hexameter", _fmt(re.findall(r"LSS|LL", pattern[:-2]) + [pattern[-2:]])
    m = PENTAMETER.match(pattern)
    if m:
        return "pentameter", _fmt(m.groups()[:3]) + " ‖ " + _fmt(m.groups()[3:])
    return None, " ".join(MARK["long" if c == "L" else "short"] for c in pattern)


def _scan_neural(name: str, parsed: list[list[list[str]]]):
    """Labels via the original predict_given_set(); confidences from the softmax."""
    m = MODELS[name]
    given_set = [[(s, None) for s in words_to_sequence(w)] for w in parsed]
    y_pred = m.predict_given_set(given_set, m.model)
    probs = m.model.last
    out = []
    for li, words in enumerate(parsed):
        # pad_sequences truncates from the front when a line exceeds max_sentence_length
        pos, line = -max(0, len(given_set[li]) - m.max_sentence_length), []
        for wi, syls in enumerate(words):
            pos += wi > 0  # skip the space token between words
            word = []
            for s in syls:
                if pos >= 0:
                    label = str(m.LABELS[y_pred[li][pos]])
                    conf = float(probs[li][pos][y_pred[li][pos]])
                else:  # longer than any training line: dropped by pad_sequences
                    label, conf = "?", 0.0
                word.append((s, label, conf, s in m.word2idx))
                pos += 1
            line.append(word)
        out.append(line)
    return out


def _scan_crf(parsed: list[list[list[str]]]):
    """Features from the original Latin_CRF.sentence_to_features (spaces removed,
    as in all of Latin_CRF's routines); Viterbi decoding of the trained weights."""
    out = []
    for words in parsed:
        sent = [(s, None) for w in words for s in w]
        labels, marg = CRF_TAGGER.predict_marginals_and_path(CRF.sentence_to_features(sent)) if sent else ([], [])
        k, line = 0, []
        for syls in words:
            word = []
            for s in syls:
                li = CRF_TAGGER.labels.index(labels[k])
                word.append((s, labels[k], float(marg[k][li]), f"0:syllable:{s}" in CRF_TAGGER.state))
                k += 1
            line.append(word)
        out.append(line)
    return out


def scan_lines(text: str, model: str = "transformer_medium") -> list[dict]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    parsed = [line_to_words(l) for l in lines]
    scanned = _scan_crf(parsed) if model == "crf" else _scan_neural(model, parsed)
    results = []
    for line, words in zip(lines, scanned):
        pattern = "".join("L" if s[1] == "long" else "S" for w in words for s in w if s[1] in ("long", "short"))
        meter, feet = fit_meter(pattern)
        results.append({
            "line": line,
            "words": [[{"syllable": s, "label": l, "confidence": round(c, 3), "known_syllable": v} for s, l, c, v in w] for w in words],
            "pattern": pattern,
            "meter": meter,
            "feet": feet,
        })
    return results


def render(results: list[dict]) -> str:
    if not results:
        return "<p class='ls-empty'>Enter a line of Latin verse above.</p>"
    rows = []
    for r in results:
        words = []
        for w in r["words"]:
            syls = "".join(
                f"<span class='ls-syl ls-{s['label']}{' ls-unsure' if s['confidence'] < 0.8 else ''}{' ls-oov' if not s['known_syllable'] else ''}' "
                f"title='{s['label']} · {s['confidence']:.0%}{'' if s['known_syllable'] else ' · unseen syllable'}'>"
                f"<span class='ls-mark'>{MARK.get(s['label'], '?')}</span><span class='ls-text'>{html.escape(s['syllable'])}</span></span>"
                for s in w
            )
            words.append(f"<span class='ls-word'>{syls}</span>")
        badge = (f"<span class='ls-badge ls-ok'>{r['meter']}</span>" if r["meter"]
                 else "<span class='ls-badge ls-no'>no dactylic fit</span>")
        rows.append(
            f"<div class='ls-line'><div class='ls-words'>{''.join(words)}</div>"
            f"<div class='ls-meta'>{badge}<span class='ls-feet'>{r['feet']}</span></div></div>"
        )
    return "".join(rows)


def scan_json(text: str, model: str = "transformer_medium") -> str:
    """Entry point called from index.html.
    model: 'transformer_tiny' | 'transformer_base' | 'transformer_medium' | 'lstm' | 'crf'."""
    results = scan_lines(text, model)
    return json.dumps({"html": render(results), "results": results}, ensure_ascii=False)
