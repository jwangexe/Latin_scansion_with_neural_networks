from keras import layers, models, ops

# Third party library imports
import numpy as np
import tensorflow as tf

# Local imports
from lsnn import utilities as util

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



# Based on https://www.depends-on-the-definition.com/guide-sequence-tagging-neural-networks-python/

class Latin_Transformer():
    """This class provides the user with tools to create an LSTM model for scanning Latin. On startup,
    it prepares the padding procedures and creates syllable and label dictionaries for the one-hot encoding
    used by the LSTM. Having done that, the class can create and save models and predict custom syllable-label lists.
    """    
    # class constants    
    PADDING = 'padding'
    LABELS = np.array(['long', 'short', 'elision', 'space', PADDING])

    def __init__(self,
        sequence_labels_folder: str,
        models_save_folder: str,
        anceps_label: bool = False
        ):

        self.anceps_label = anceps_label
        self.sequence_labels_folder = sequence_labels_folder
        self.models_save_folder = models_save_folder

        if self.anceps_label:
            self.LABELS = np.array(['long', 'short', 'elision', 'space', self.PADDING, 'anceps'])

        # To make the LSTM with integer hashing working, we need to make a list of all syllables from all the texts we are looking at 
        # First, find all our pickle files   
        all_sequence_label_pickles = util.create_files_list(self.sequence_labels_folder, 'pickle') 
        # Merge them into one big file list
        sequence_labels_all_set = util.merge_sequence_label_lists(all_sequence_label_pickles, self.sequence_labels_folder) 
        # Retrieve all syllables from this big list
        all_text_syllables = self.retrieve_syllables_from_sequence_label_list(sequence_labels_all_set)
        # We need to extract the max sentence length over all these texts to get the padding correct later
        self.max_sentence_length = self.retrieve_max_sentence_length(sequence_labels_all_set)       
        # And we need to create a list of all unique syllables for our word2idx one-hot encoding
        self.unique_syllables = np.append(sorted(list(set(all_text_syllables))), self.PADDING)
        self.word2idx, self.label2idx = self.create_idx_dictionaries(self.unique_syllables, self.LABELS)            

        # With that out of the way, we can start the LSTM process
        print('Preprocessing done.')
        print('number of syllables: ', len(self.unique_syllables))
        print('number of labels: ', len(self.LABELS))
        print('max_sentence_length: ', self.max_sentence_length)

    def predict_given_set(self, 
        given_set: list,
        model
        ) -> list:
        """Given the set and the model, this function returns a list with predictions.

        Args:
            given_set (list): with sequences to be predicted
            model (tensorflow model): that has to predict the given set 

        Returns:
            list: with predictions per given syllable in the given_set
        """        
        # now we map the sentences and labels to a sequence of numbers
        X = [[self.word2idx[w[0]] for w in s] for s in given_set]  # key 0 are labels
        # and then (post)pad the sequences using the PADDING label.
        X = tf.keras.utils.pad_sequences(
                maxlen = self.max_sentence_length, 
                sequences = X, 
                padding = "post", 
                value = self.word2idx[self.PADDING]
                ) # value is our padding key

        y_pred = model.predict(X)

        if self.anceps_label: # we dont want to predict the label anceps, so we delete it from the possible predictions
            for line in y_pred:
                for syllable in line:
                    # check which label has the highest confidence
                    position = np.where(syllable == np.amax(syllable))
                    # if it is the anceps label, check confidence for long and short
                    if position[0][0] == 5:
                        long = syllable[0]
                        short = syllable[1]
                        if long > short:
                            syllable = np.array([1, 0, 0, 0, 0, 0])
                        else:
                            syllable = np.array([0, 1, 0, 0, 0, 0])      
        else:
            y_pred = np.argmax(y_pred, axis=-1)

        return y_pred

    def create_model(self, 
        text : str, 
        num_epochs : int,
        save_model : bool = True,
        model_name : str = 'default'
        ):
        """Function to create a model given a text (pickled sequence label file). Returns the
        created model. In addition, allows the model to be saved to disk.

        Args:
            text (string): name of pickled sequence label file to be used for training
            save_model (bool, optional): whether the created model is to be saved to disk. Defaults to True.
            model_name (str, optional): name to be given to the saved model (folder name). Defaults to 'default'.

        Returns:
            tensorflow model: containing the trained LSTM
        """        
        sequence_labels_training_text = util.pickle_read(self.sequence_labels_folder, text)
        
        X_train, y_train = self.create_X_y_sets(
            sequence_labels_training_text, 
            self.word2idx, 
            self.label2idx, 
            self.max_sentence_length
        )

        model = self.get_model()

        model, history = self.fit_model(
            model = model,
            X = X_train,
            y = y_train,
            batch_size = 32,
            epochs = num_epochs,
            split = 0.2,
            verbose = True
        )   

        if save_model: 
            path = self.models_save_folder + model_name + ".keras" # get model path name for saving or loading
            model.save(path)

        return model, history
     

    def get_model(self, embed_dim=32, num_heads=2, ff_dim=32):
        """
        Build a Transformer-based classification model.

        Args:
            d_model: Integer, the dimension of the Transformer embeddings.
            num_heads: Integer, the number of attention heads.
            ff_dim: Integer, the dimension of the feed-forward network.
            num_layers: Integer, the number of Transformer layers.

        Returns:
            A compiled Keras model.
        """

        # Define input layer
        inputs = layers.Input(shape=(self.max_sentence_length,))

        # Define and add embedding layer
        embedding_layer = TokenAndPositionEmbedding(self.max_sentence_length, len(self.unique_syllables), embed_dim)
        x = embedding_layer(inputs)

        # Define and add Transformer block (including attention, normalisation and dropout layers)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D(keepdims=True, data_format="channels_first")(x)
        #x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)

        # Define and add output layer
        outputs = layers.Dense(len(self.LABELS), activation="softmax")(x)

        # Compile model
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        return model

    """
    # Define the Transformer model
    def get_model(self, d_model=128, num_heads=4, ff_dim=256, num_layers=2):
        
        inputs = layers.Input(shape=(self.max_sentence_length,))  # Input shape is (29,)

        # add embedding layer

        # Expand dimensions to make it compatible with Transformer (add sequence dimension)
        x = layers.Reshape((self.max_sentence_length, 1))(inputs)  # Reshape to (29, 1)

        # Dense layer to project input to d_model dimensions
        x = layers.Dense(d_model)(x)

        # Transformer Encoder Layers
        for _ in range(num_layers):
            # Multi-Head Attention
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)

            # Feed-Forward Network
            ffn = layers.Dense(ff_dim, activation='relu')(attention_output)
            ffn = layers.Dense(d_model)(ffn)
            x = layers.LayerNormalization(epsilon=1e-6)(ffn + attention_output)

        # Global average pooling to reduce sequence dimension
        x = layers.GlobalAveragePooling1D()(x)

        # Output Layer for Classification
        outputs = layers.Dense(len(self.LABELS), activation='softmax')(x)

        # Compile Model
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    """

    def fit_model(self, 
        model,
        X: list,
        y: list,
        batch_size: int,
        epochs: int,
        split: float,
        verbose: bool
        ):

        history = model.fit(
            X, 
            y, 
            batch_size = batch_size, 
            epochs = epochs, 
            validation_split = split, 
            verbose = verbose
        )
        
        return model, history

    def load_model(self, path: str):
        """Loads the model from the given path

        Args:
            path (string): containing the path of the saved model

        Returns:
            tf.keras model: used to scan poetry
        """        
        return tf.keras.models.load_model(path)

    def flatten_list(self, given_list: list) -> list:
        """Flattens the given list. Meaning that nested lists inside the given lists are
        turned into one list. For example, [[a,b],[c]] -> [a,b,c]

        Args:
            given_list (list): nested list that needs to be flattened

        Returns:
            list: flattened list
        """        
        return [item for sublist in given_list for item in sublist]

    def retrieve_max_sentence_length(self, sequence_labels: list) -> int:
        """Returns the maximum sentence length of the given sequence label list. Used for padding calculations

        Args:
            sequence_labels (list): with sentences and their syllables and labels

        Returns:
            int: of maximum sentence length
        """        
        max_len : int = 0

        for sentence in sequence_labels:
            if len(sentence) > max_len:
                max_len = len(sentence)

        return max_len

    def retrieve_syllables_from_sequence_label_list(self, sequence_labels: list) -> list:
        """returns all the syllables from the given sequence label list 

        Args:
            sequence_labels (list): with sequence labels

        Returns:
            list: of all syllables in the given texts
        """
        syllable_list = []

        for sentence in sequence_labels:
            for syllable, label in sentence:
                syllable_list.append(syllable)

        return syllable_list

    def create_idx_dictionaries(self,
        unique_syllables: list, 
        labels: list):
        """This function creates the idx dictionaries needed for creating an LSTM on syllable hashes

        Args:
            unique_syllables (list): of unique syllables
            labels (list): of unique syllables

        Returns:
            dict: of syllable -> hash and label -> hash
        """        
        word2idx = {w: i for i, w in enumerate(unique_syllables)}
        label2idx = {t: i for i, t in enumerate(labels)}   
        return word2idx, label2idx     

    def create_X_y_sets(self, 
                        given_set: list, 
                        word2idx: list, 
                        label2idx: list, 
                        max_sentence_length: int):
        """Creates X and y sets that can be used by LSTM. Pads sequences and converts y to categorical

        Args:
            given_set (list): set to be converted to X and y
            word2idx (dict): with hashes for our syllables
            label2idx (dict): with hashes for our labels
            max_sentence_length (int): max length of a sentence: used for padding

        Returns:
            list: of X set
            list: of y set
        """        
        # now we map the sentences and labels to a sequence of numbers
        X = [[word2idx[w[0]] for w in s] for s in given_set]  # key 0 are labels
        y = [[label2idx[w[1]] for w in s] for s in given_set] # key 1 are labels
        # and then (post)pad the sequences using the PADDING label.
        X = tf.keras.utils.pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx[self.PADDING]) # value is our padding key
        y = tf.keras.utils.pad_sequences(maxlen=max_sentence_length, sequences=y, padding="post", value=label2idx[self.PADDING])
        # for training the network we also need to change the labels to categorial.
        y = np.array([tf.keras.utils.to_categorical(i, num_classes=len(self.LABELS)) for i in y])

        print(X.shape)
        print(y.shape)

        return X, y

# if __name__ == "__main__":

#     lstm = Latin_LSTM(
#         sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
#         models_save_folder = './models/lstm/',
#         anceps_label = False,
#         ) 

#     model = lstm.create_model(
#         num_epochs = 2,
#         text = 'HEX_ELE-all.pickle', 
#         save_model = True, 
#         model_name = 'temp'
#         )

        
