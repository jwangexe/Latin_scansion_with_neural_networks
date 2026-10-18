"""Turns typed Latin verse into the syllable sequences the models were trained on.

Syllabification mirrors the pipeline used to build the training data
(BNagy's MQDQ syllabifier, with its u -> v consonant fix-up), so syllables
typed by a user land on the same vocabulary the model was trained on.
"""
import re

from cltk_hax.syllabifier import Syllabifier

SPACE = "-"  # word-boundary token, as in the Pedecerto / Anceps parsers

_syllabify = Syllabifier().syllabify
_VOWELS = set("aeiouy")


def _consonantal_u(word: str) -> str:
    """Same u->v rule MQDQ applies: initial u before a vowel, or VuV."""
    chars = list(word)
    for i, c in enumerate(chars):
        if c != "u" or i == len(chars) - 1:
            continue
        nxt = chars[i + 1]
        if nxt in _VOWELS - {"u"} and (i == 0 or chars[i - 1] in _VOWELS):
            chars[i] = "v"
    return "".join(chars)


def _syllabify_word(word: str) -> list[str]:
    """Consonantal i follows the training data: word-initial i before a vowel
    is written j (jam, ju.ve.nis); intervocalic i closes the previous
    syllable (troi.ae, ei.us)."""
    word = _consonantal_u(word)
    if len(word) > 1 and word[0] == "i" and word[1] in _VOWELS:
        word = "j" + word[1:]
    word = re.sub(r"(?<![qg]u)(?<=[aeiouy])i(?=[aeouy])", "j", word)
    syls = [s.lower() for s in _syllabify(word)]
    # ae, oe, au are diphthongs far more often than not
    k = 1
    while k < len(syls):
        prev, cur = syls[k - 1], syls[k]
        if prev[-1] + cur[0] in ("ae", "oe", "au"):
            syls[k - 1], syls[k] = prev + cur[0], cur[1:]
            # leftover consonants (a.es -> aes) stay with the diphthong
            if not any(c in _VOWELS for c in syls[k]):
                syls[k - 1] += syls.pop(k)
                continue
        k += 1
    for k in range(1, len(syls)):
        if syls[k].startswith("j") and syls[k - 1][-1] in _VOWELS and len(syls[k]) > 1:
            syls[k - 1] += "i"
            syls[k] = syls[k][1:]
    return syls


def line_to_words(line: str) -> list[list[str]]:
    words = re.findall(r"[a-z]+", line.lower())
    return [_syllabify_word(w) for w in words]


def words_to_sequence(words: list[list[str]]) -> list[str]:
    seq = []
    for i, syls in enumerate(words):
        if i:
            seq.append(SPACE)
        seq.extend(syls)
    return seq
