"""Compatibility layer so the ORIGINAL repo code (lsnn/crf.py, lsnn/lstm.py,
transformer.py) can run inference in the browser under Pyodide.

TensorFlow, Keras, pandas and python-crfsuite are not available in Pyodide, so:
  * install_stubs() registers placeholder modules so the original files import;
  * the trained Keras models are replaced by NumPy re-implementations of the same
    forward pass (NumpyLSTM, NumpyTransformer) with a Keras-style .predict();
  * the trained CRF's weights are decoded with Viterbi (CRFTagger), which is what
    crfsuite does at prediction time.
Training never touches this file: it used the original code with real TF / crfsuite.
"""
import json
import sys
import types

import numpy as np


# --------------------------------------------------------------------------- stubs
def _pad_sequences(sequences, maxlen, padding="pre", value=0, **_):
    """Same behaviour as tf.keras.utils.pad_sequences for the arguments the repo uses
    (truncating='pre' is Keras' default)."""
    out = np.full((len(sequences), maxlen), value, dtype="int32")
    for i, s in enumerate(sequences):
        s = list(s)[-maxlen:]
        if padding == "post":
            out[i, : len(s)] = s
        else:
            out[i, maxlen - len(s):] = s
    return out


def install_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules.setdefault(name, m)
        return sys.modules[name]

    class _Layer:  # transformer.py subclasses keras.layers.Layer at import time
        def __init__(self, *a, **k):
            pass

    layers = mod("keras.layers", Layer=_Layer)
    mod("keras", layers=layers, models=mod("keras.models"), ops=mod("keras.ops"))
    utils = mod("tensorflow.keras.utils", pad_sequences=_pad_sequences)
    mod("tensorflow", keras=mod("tensorflow.keras", utils=utils))
    mod("pandas")
    mod("sklearn", model_selection=mod("sklearn.model_selection"))
    mod("sklearn_crfsuite", metrics=mod("sklearn_crfsuite.metrics"))


# ------------------------------------------------------------------- Keras in NumPy
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(x, gamma, beta, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def _lstm(x, kernel, recurrent, bias):
    """keras.layers.LSTM, return_sequences=True, gate order i, f, c, o.
    x: (batch, time, features) -> (batch, time, units)."""
    units = recurrent.shape[0]
    h = np.zeros((x.shape[0], units))
    c = np.zeros_like(h)
    xz = x @ kernel + bias
    hs = []
    for t in range(x.shape[1]):
        z = xz[:, t] + h @ recurrent
        i, f, g, o = (z[:, k * units:(k + 1) * units] for k in range(4))
        c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
        h = _sigmoid(o) * np.tanh(c)
        hs.append(h)
    return np.stack(hs, axis=1)


class NumpyLSTM:
    """Forward pass of Latin_LSTM.get_model(): Embedding -> Bidirectional(LSTM) ->
    Dense(softmax). No masking, exactly like the original (padding is processed)."""

    def __init__(self, path):
        self.w = dict(np.load(path))

    def predict(self, X, **_):
        w = self.w
        x = w["embedding"][X]
        fwd = _lstm(x, w["fwd_kernel"], w["fwd_recurrent"], w["fwd_bias"])
        bwd = _lstm(x[:, ::-1], w["bwd_kernel"], w["bwd_recurrent"], w["bwd_bias"])[:, ::-1]
        self.last = _softmax(np.concatenate([fwd, bwd], -1) @ w["dense_kernel"] + w["dense_bias"])
        return self.last


class NumpyTransformer:
    """Forward pass of Latin_Transformer.get_model(): TokenAndPositionEmbedding ->
    TransformerBlock x num_blocks -> GlobalAveragePooling1D(keepdims, channels_first)
    -> Dense(20, relu) -> Dense(softmax). Dropout is inactive at inference."""

    def __init__(self, path):
        self.w = dict(np.load(path))
        self.num_blocks = int(self.w["num_blocks"])

    @staticmethod
    def _block(x, w, p):
        # keras.layers.MultiHeadAttention(num_heads, key_dim), self-attention, no mask
        q = np.einsum("btd,dhk->bhtk", x, w[p + "q_kernel"]) + w[p + "q_bias"][None, :, None]
        k = np.einsum("btd,dhk->bhtk", x, w[p + "k_kernel"]) + w[p + "k_bias"][None, :, None]
        v = np.einsum("btd,dhk->bhtk", x, w[p + "v_kernel"]) + w[p + "v_bias"][None, :, None]
        att = _softmax(np.einsum("bhqk,bhsk->bhqs", q / np.sqrt(q.shape[-1]), k))
        ctx = np.einsum("bhqs,bhsk->bhqk", att, v)
        attn_out = np.einsum("bhtk,hkd->btd", ctx, w[p + "o_kernel"]) + w[p + "o_bias"]
        out1 = _layer_norm(x + attn_out, w[p + "ln1_gamma"], w[p + "ln1_beta"])
        ffn = np.maximum(out1 @ w[p + "ffn1_kernel"] + w[p + "ffn1_bias"], 0) @ w[p + "ffn2_kernel"] + w[p + "ffn2_bias"]
        return _layer_norm(out1 + ffn, w[p + "ln2_gamma"], w[p + "ln2_beta"])

    def predict(self, X, **_):
        w = self.w
        x = w["tok_emb"][X] + w["pos_emb"][np.arange(X.shape[1])]
        for i in range(self.num_blocks):
            x = self._block(x, w, f"b{i}_")
        x = x.mean(-1, keepdims=True)  # channels_first pooling averages the feature axis
        x = np.maximum(x @ w["dense1_kernel"] + w["dense1_bias"], 0)
        self.last = _softmax(x @ w["dense2_kernel"] + w["dense2_bias"])
        return self.last


# ----------------------------------------------------------------------------- CRF
class CRFTagger:
    """Viterbi decoding with the weights of the sklearn_crfsuite.CRF trained by
    Latin_CRF.fit_model(). Feature dicts are flattened the way crfsuite does:
    string values become 'key:value' (weight 1), numbers/bools keep 'key'."""

    def __init__(self, path):
        d = json.load(open(path))
        self.labels = d["labels"]
        li = {l: i for i, l in enumerate(self.labels)}
        self.state = {}
        for attr, label, wt in d["state_features"]:
            self.state.setdefault(attr, np.zeros(len(self.labels)))[li[label]] = wt
        self.trans = np.zeros((len(self.labels), len(self.labels)))
        for a, b, wt in d["transition_features"]:
            self.trans[li[a], li[b]] = wt

    def _emissions(self, feats):
        E = np.zeros((len(feats), len(self.labels)))
        for t, f in enumerate(feats):
            for key, val in f.items():
                if isinstance(val, str):
                    attr, val = f"{key}:{val}", 1.0
                else:
                    attr, val = key, float(val)
                if attr in self.state:
                    E[t] += val * self.state[attr]
        return E

    def predict_marginals_and_path(self, feats):
        E = self._emissions(feats)
        n, L = E.shape
        score, back = E[0].copy(), np.zeros((n, L), dtype=int)
        for t in range(1, n):
            cand = score[:, None] + self.trans + E[t][None, :]
            back[t] = cand.argmax(0)
            score = cand.max(0)
        path = [int(score.argmax())]
        for t in range(n - 1, 0, -1):
            path.append(int(back[t][path[-1]]))
        path = path[::-1]
        # forward-backward marginals for a per-syllable confidence
        alpha = np.zeros((n, L)); beta = np.zeros((n, L))
        alpha[0] = E[0]
        for t in range(1, n):
            alpha[t] = E[t] + np.logaddexp.reduce(alpha[t - 1][:, None] + self.trans, axis=0)
        for t in range(n - 2, -1, -1):
            beta[t] = np.logaddexp.reduce(self.trans + (E[t + 1] + beta[t + 1])[None, :], axis=1)
        logz = np.logaddexp.reduce(alpha[-1])
        marg = np.exp(alpha + beta - logz)
        return [self.labels[i] for i in path], marg

    def predict(self, X):
        return [self.predict_marginals_and_path(f)[0] for f in X]
