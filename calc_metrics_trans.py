import numpy as np

from transformer import Latin_Transformer
from lsnn import config as conf
from lsnn import utilities as util

from sklearn.metrics import confusion_matrix, f1_score

tran = Latin_Transformer(
    sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
    models_save_folder = conf.MODELS_SAVE_FOLDER,
    anceps_label = False,
)

print(tran.label2idx)

test_data = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
ypred = np.load("trans_result.npy")

print(ypred.shape)

y_test = np.full(shape=ypred.shape, fill_value=tran.label2idx["padding"], dtype=np.int64)
for i in range(len(test_data)):
    for j in range(len(test_data[i])):
        label = test_data[i][j][1]
        y_test[i, j] = tran.label2idx[label]


print(ypred)
print(y_test)

print("="*28)
print("Confusion Matrix:")
print(confusion_matrix(y_test.flatten(), ypred.flatten(), labels=[0, 1, 2]))

f1scores = f1_score(y_test.flatten(), ypred.flatten(), labels=[0, 1, 2], average=None)
header = "|{:^8}|{:^8}|{:^8}|"

print("-"*28)
print(header.format("Long", "Short", "Elision"))
print("="*28)

format = "|{:^8.4f}|{:^8.4f}|{:^8.4f}|"
print(format.format(*f1scores))
print("-"*28)