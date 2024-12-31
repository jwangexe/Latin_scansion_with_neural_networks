from lsnn import dataset_creation
from lsnn.lstm import Latin_LSTM
from lsnn.crf import Latin_CRF

from lsnn import config as conf
from lsnn import utilities as util

import numpy as np
import matplotlib.pyplot as plt

from transformer import Latin_Transformer

####################
# DATASET CREATION #
####################
dataset_creation.Pedecerto_parser().convert_pedecerto_xml_to_syllable_sequence_files(
    input_folder = conf.PEDECERTO_SCANSION_FOLDER,
    output_folder = conf.SEQUENCE_LABELS_FOLDER    
)

dataset_creation.Anceps_parser().convert_anceps_json_to_syllable_sequence_files(
    input_folder = conf.ANCEPS_SCANSION_FOLDER,
    output_folder = conf.SEQUENCE_LABELS_FOLDER
)

util.combine_sequence_label_lists(
    list_with_file_names = util.create_files_list(conf.SEQUENCE_LABELS_FOLDER, 'pickle'), 
    output_name = 'combined.txt', 
    destination_path = conf.SEQUENCE_LABELS_FOLDER,
    add_extension = False
)

# ##############
# # LSTM MODEL #
# ##############
# lstm = Latin_LSTM(
#     sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
#     models_save_folder = conf.MODELS_SAVE_FOLDER,
#     anceps_label = False,
# ) 

# model, history = lstm.create_model(
#     text = 'HEX_ELE-all.pickle', 
#     num_epochs = 1,
#     save_model = True, 
#     model_name = 'temp'
# )

# model = lstm.load_model(
#     path = conf.MODELS_SAVE_FOLDER + 'temp.keras'
# )

# # with the model, we can predict the labels of a given set
# test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
# lstm_result = lstm.predict_given_set(test_set, model)


#####################
# TRANSFORMER MODEL #
#####################
transformer = Latin_Transformer(
    sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
    models_save_folder = conf.MODELS_SAVE_FOLDER,
    anceps_label = False,
)

model, history = transformer.create_model(
    text = 'HEX_ELE-all.pickle', 
    num_epochs = 25,
    save_model = True, 
    model_name = 'temp'
)

model = transformer.load_model(
    path = conf.MODELS_SAVE_FOLDER + 'temp.keras'
)

test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
trans_result = transformer.predict_given_set(test_set, model)


def plot_metrics_epoch(history):
    # Extract training and validation metrics
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_loss) + 1)

    # Plot the metrics
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'o-', label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, 'o-', label='Validation Loss', color='red')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, 'o-', label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracy, 'o-', label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Add a main title for the entire figure
    plt.suptitle('Model Training Metrics', fontsize=16)

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_metrics_epoch(history)

# "result" is the predicted labels
# TODO: compare with actual test labels to get metrics
# print(f"Test data: {test_set[:5]}")
# print(f"LSTM result: {lstm_result}")

# np.save("lstm_result.npy", lstm_result)
np.save("trans_result.npy", trans_result)