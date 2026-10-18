# Trains the four TransScan Transformer sizes (paper, Table 3) with the original
# Latin_Transformer.create_model routine from transformer.py, then scores each one
# the way calc_metrics_trans.py does. transformer.py itself is used unchanged:
# its get_model() builds one TransformerBlock, so the subclass below only adds a
# loop to stack `num_blocks` of the same block. Run from the repo root:
#   python hf_space/build/train_transformers.py tiny   (or base / medium / large)
# An optional second argument fixes the random seed, e.g. `large 2`.
import sys; sys.path.insert(0, '.')
import json

import keras
import numpy as np
from keras import layers, models
from sklearn.metrics import f1_score

from transformer import Latin_Transformer, TransformerBlock, TokenAndPositionEmbedding
from lsnn import config as conf
from lsnn import utilities as util

# Table 3: Num. blocks, Num. heads, Emb-Dim, Ffd-Dim
SIZES = {
    'tiny':   dict(num_blocks=1, num_heads=4, embed_dim=32, ff_dim=32),
    'base':   dict(num_blocks=2, num_heads=4, embed_dim=48, ff_dim=48),
    'medium': dict(num_blocks=3, num_heads=6, embed_dim=48, ff_dim=48),
    'large':  dict(num_blocks=4, num_heads=6, embed_dim=64, ff_dim=64),
}


class Sized_Transformer(Latin_Transformer):
    def __init__(self, num_blocks, num_heads, embed_dim, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.size = dict(num_blocks=num_blocks, num_heads=num_heads, embed_dim=embed_dim, ff_dim=ff_dim)

    def get_model(self):
        # Same layers as Latin_Transformer.get_model(), with the block repeated.
        s = self.size
        inputs = layers.Input(shape=(self.max_sentence_length,))
        x = TokenAndPositionEmbedding(self.max_sentence_length, len(self.unique_syllables), s['embed_dim'])(inputs)
        for _ in range(s['num_blocks']):
            x = TransformerBlock(s['embed_dim'], s['num_heads'], s['ff_dim'])(x)
        x = layers.GlobalAveragePooling1D(keepdims=True, data_format="channels_first")(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(len(self.LABELS), activation="softmax")(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model


name = sys.argv[1]
tag = name
if len(sys.argv) > 2:
    keras.utils.set_random_seed(int(sys.argv[2]))
    tag = f'{name}_seed{sys.argv[2]}'
t = Sized_Transformer(**SIZES[name], sequence_labels_folder=conf.SEQUENCE_LABELS_FOLDER,
                      models_save_folder=conf.MODELS_SAVE_FOLDER, anceps_label=False)
model, history = t.create_model(text='HEX_ELE-all.pickle', num_epochs=25, save_model=True,
                                model_name=f'transformer_{tag}')

# calc_metrics_trans.py
test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
ypred = t.predict_given_set(test_set, model)
np.save(f"trans_result_{tag}.npy", ypred)
y_test = np.full(shape=ypred.shape, fill_value=t.label2idx["padding"], dtype=np.int64)
for i in range(len(test_set)):
    for j in range(len(test_set[i])):
        y_test[i, j] = t.label2idx[test_set[i][j][1]]
f1 = f1_score(y_test.flatten(), ypred.flatten(), labels=[0, 1, 2], average=None)
print(tag, "F1 long/short/elision:", f1.round(4))
json.dump({'size': SIZES[name], 'f1': dict(zip(['long', 'short', 'elision'], map(float, f1))),
           'history': {k: list(map(float, v)) for k, v in history.history.items()}},
          open(f'hf_space/build/metrics_{tag}.json', 'w'))
