# Trains the CRF with the original routine from lsnn/crf.py on HEX_ELE-all and
# exports its learned weights so the browser can run the same model.
# Run from the repo root:  python hf_space/build/train_crf.py
import sys; sys.path.insert(0, '.')
import json
import pickle

from lsnn.crf import Latin_CRF
from lsnn import config as conf
from lsnn import utilities as util

latin_crf = Latin_CRF()

sequence_labels = latin_crf.remove_space_from_syllable_sequence(
    util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle'))
X, y = latin_crf.convert_text_to_feature_sets(sequence_labels)
crf = latin_crf.fit_model(X, y)

json.dump({
    'labels': list(crf.classes_),
    'state_features': [[attr, label, w] for (attr, label), w in crf.state_features_.items()],
    'transition_features': [[a, b, w] for (a, b), w in crf.transition_features_.items()],
}, open('hf_space/model/crf.json', 'w'))
pickle.dump(crf, open('hf_space/build/crf.pkl', 'wb'))  # kept locally for export.py's check
print('attributes sample:', list(crf.state_features_)[:8])
print('labels:', crf.classes_)
