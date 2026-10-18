# Exports the trained models for the browser and checks that the browser-side
# re-implementations reproduce the originals on every line of HEX_ELE-all.
# Run from the repo root after train_lstm.py and train_crf.py:
#   python hf_space/build/export.py [large_run]   e.g. large_seed2 (default: large)
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'hf_space')
import json
import os
import pickle

import keras
import numpy as np

from lsnn.lstm import Latin_LSTM
from lsnn.crf import Latin_CRF
from lsnn import config as conf
from lsnn import utilities as util
from transformer import Latin_Transformer, TransformerBlock, TokenAndPositionEmbedding
from compat import NumpyLSTM, NumpyTransformer, CRFTagger

OUT = 'hf_space/model/'
args = dict(sequence_labels_folder=conf.SEQUENCE_LABELS_FOLDER, models_save_folder=conf.MODELS_SAVE_FOLDER, anceps_label=False)
lstm, trans = Latin_LSTM(**args), Latin_Transformer(**args)
assert lstm.word2idx == trans.word2idx and lstm.max_sentence_length == trans.max_sentence_length
json.dump({'unique_syllables': list(map(str, lstm.unique_syllables)), 'max_sentence_length': int(lstm.max_sentence_length)},
          open(OUT + 'vocab.json', 'w'))

# ---- LSTM
m = keras.models.load_model(conf.MODELS_SAVE_FOLDER + 'lstm_HEX_ELE-all.keras')
W = {v.path: np.asarray(v.numpy()) for v in m.weights}
pre = ''
np.savez_compressed(OUT + 'lstm_weights.npz',
    embedding=W['embedding/embeddings'],
    fwd_kernel=W[pre + 'forward_lstm/lstm_cell/kernel'], fwd_recurrent=W[pre + 'forward_lstm/lstm_cell/recurrent_kernel'], fwd_bias=W[pre + 'forward_lstm/lstm_cell/bias'],
    bwd_kernel=W[pre + 'backward_lstm/lstm_cell/kernel'], bwd_recurrent=W[pre + 'backward_lstm/lstm_cell/recurrent_kernel'], bwd_bias=W[pre + 'backward_lstm/lstm_cell/bias'],
    dense_kernel=W['dense/kernel'], dense_bias=W['dense/bias'])
lstm_keras = m

# ---- Transformers (TransScan sizes, trained by train_transformers.py)
SIZES = ['tiny', 'base', 'medium', 'large']
RUNS = {s: s for s in SIZES}
if len(sys.argv) > 1:
    RUNS['large'] = sys.argv[1]
RUNS = {s: r for s, r in RUNS.items() if os.path.exists(conf.MODELS_SAVE_FOLDER + f'transformer_{r}.keras')}


def export_transformer(m, path):
    v = lambda var: np.asarray(var.numpy())
    emb = next(l for l in m.layers if isinstance(l, TokenAndPositionEmbedding))
    blocks = [l for l in m.layers if isinstance(l, TransformerBlock)]
    dense = [l for l in m.layers if isinstance(l, keras.layers.Dense)]
    W = {'num_blocks': np.array(len(blocks)), 'tok_emb': v(emb.token_emb.embeddings), 'pos_emb': v(emb.pos_emb.embeddings),
         'dense1_kernel': v(dense[0].kernel), 'dense1_bias': v(dense[0].bias),
         'dense2_kernel': v(dense[1].kernel), 'dense2_bias': v(dense[1].bias)}
    for i, b in enumerate(blocks):
        p, att = f'b{i}_', b.att
        for n, sub in [('q', att._query_dense), ('k', att._key_dense), ('v', att._value_dense), ('o', att._output_dense)]:
            W[p + n + '_kernel'], W[p + n + '_bias'] = v(sub.kernel), v(sub.bias)
        f1, f2 = b.ffn.layers
        W[p + 'ffn1_kernel'], W[p + 'ffn1_bias'], W[p + 'ffn2_kernel'], W[p + 'ffn2_bias'] = v(f1.kernel), v(f1.bias), v(f2.kernel), v(f2.bias)
        W[p + 'ln1_gamma'], W[p + 'ln1_beta'] = v(b.layernorm1.gamma), v(b.layernorm1.beta)
        W[p + 'ln2_gamma'], W[p + 'ln2_beta'] = v(b.layernorm2.gamma), v(b.layernorm2.beta)
    np.savez_compressed(path, **W)


custom = {'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding}
trans_keras = {}
for size, run in RUNS.items():
    trans_keras[size] = keras.models.load_model(conf.MODELS_SAVE_FOLDER + f'transformer_{run}.keras', custom_objects=custom)
    export_transformer(trans_keras[size], OUT + f'transformer_{size}_weights.npz')

# ---- Check: original predict_given_set with the Keras model vs. the NumPy model, every line
test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
checks = [('LSTM', lstm, lstm_keras, NumpyLSTM(OUT + 'lstm_weights.npz'))]
checks += [(f'Transformer {s}', trans, trans_keras[s], NumpyTransformer(OUT + f'transformer_{s}_weights.npz')) for s in RUNS]
for name, cls, km, nm in checks:
    a = cls.predict_given_set(test_set, km)
    b = cls.predict_given_set(test_set, nm)
    print(f'{name}: NumPy labels identical to Keras on {(a == b).mean():.6%} of positions')

# ---- Check: crfsuite vs Viterbi on every line
crf_model = pickle.load(open('hf_space/build/crf.pkl', 'rb'))
c = Latin_CRF()
X, _ = c.convert_text_to_feature_sets(c.remove_space_from_syllable_sequence(test_set))
a = [l for s in crf_model.predict(X) for l in s]
b = [l for s in CRFTagger(OUT + 'crf.json').predict(X) for l in s]
print(f'CRF: Viterbi labels identical to crfsuite on {np.mean(np.array(a) == np.array(b)):.6%} of syllables')
