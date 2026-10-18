# Trains the LSTM with the original routine from lsnn/lstm.py (same call as the
# LSTM block in test.py, with 25 epochs to match the transformer), then scores it
# the way calc_metrics_lstm.py does. Run from the repo root:
#   python hf_space/build/train_lstm.py
import sys; sys.path.insert(0, '.')
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from lsnn.lstm import Latin_LSTM
from lsnn import config as conf
from lsnn import utilities as util

lstm = Latin_LSTM(
    sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
    models_save_folder = conf.MODELS_SAVE_FOLDER,
    anceps_label = False,
)

model, history = lstm.create_model(
    text = 'HEX_ELE-all.pickle',
    num_epochs = 25,
    save_model = True,
    model_name = 'lstm_HEX_ELE-all'
)

test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
ypred = lstm.predict_given_set(test_set, model)
np.save("lstm_result.npy", ypred)

y_test = np.full(shape=ypred.shape, fill_value=lstm.label2idx["padding"], dtype=np.int64)
for i in range(len(test_set)):
    for j in range(len(test_set[i])):
        y_test[i, j] = lstm.label2idx[test_set[i][j][1]]
print(confusion_matrix(y_test.flatten(), ypred.flatten(), labels=[0, 1, 2]))
print("F1 long/short/elision:", f1_score(y_test.flatten(), ypred.flatten(), labels=[0, 1, 2], average=None).round(4))
