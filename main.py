'''
       ___       ___           ___     
      /\__\     /\  \         /\__\    
     /:/  /    /::\  \       /::|  |   
    /:/  /    /:/\ \  \     /:|:|  |   
   /:/  /    _\:\~\ \  \   /:/|:|__|__ 
  /:/__/    /\ \:\ \ \__\ /:/ |::::\__\ 
  \:\  \    \:\ \:\ \/__/ \/__/~~/:/  /
   \:\  \    \:\ \:\__\         /:/  / 
    \:\  \    \:\/:/  /        /:/  /  
     \:\__\    \::/  /        /:/  /   
      \/__/     \/__/         \/__/    
'''
# Latin Scansion Model
# Philippe Bors and Luuk Nolden
# Leiden University 2021

# Library Imports
import numpy as np
from progress.bar import Bar
import argparse

# Class Imports
# from word2vec import Word_vector_creator 
# from preprocessor import Text_preprocessor 
# from neuralnetwork import Neural_network_handler

import utilities as util

p = argparse.ArgumentParser(description='Argument parser for the LatinScansionModel')
# positional args:
p.add_argument('--pedecerto-conversion', action="store_true",
                help='converts the pedecerto XML files stored in pedecerto/xml_files into labeled dataframes')
p.add_argument('--sequence_labels_conversion', action="store_true", 
               help='converts the pedecerto dataframes stored in pedecerto/df_pedecerto into sequence labeling lists')
# p.add_argument('--dataset', default=os.getcwd() + '/pythia/data/datasets/greek_epigraphy_dict.p', type=str,
#                help='dataset file')
# p.add_argument('--dataset_test_set', default='valid', type=str, help='valid/test set split')
# p.add_argument('--load_checkpoint', default='', type=str, help='load from checkpoint')
# training behaviour options:
# p.add_argument('--batch_size', default=32, type=int, help='batch size')
# p.add_argument('--learning_rate', default=3e-4, type=float, help='learning rate')
# p.add_argument('--grad_clip', default=5., type=float, help='gradient norm clipping')
# p.add_argument('--beam_width', default=5, type=int, help='beam search width')
# p.add_argument('--eval_samples', default=1600, type=int, help='number of evaluation samples')
# p.add_argument('--summary_dir', default=os.getcwd() + '/tf/summary/', type=str, help='summary directory')
# p.add_argument('--checkpoint_dir', default=os.getcwd() + '/tf/checkpoint/', type=str, help='checkpoint directory')
# p.add_argument('--checkpoint_interval', default=500, type=int, help='checkpoint interval')
# p.add_argument('--eval_interval', default=10000, type=int, help='eval interval')
# p.add_argument('--report_interval', default=1000, type=int, help='report interval')
# p.add_argument('--training_iterations', default=1000000, type=int, help='number of training iterations')
# p.add_argument('--context_char_min', default=100, type=int, help='minimum context characters')
# p.add_argument('--context_char_max', default=1000, type=int, help='minimum context characters')
# p.add_argument('--pred_char_min', default=1, type=int, help='minimum pred characters')
# p.add_argument('--pred_char_max', default=10, type=int, help='minimum pred characters')
# p.add_argument('--missing_char_min', default=1, type=int, help='minimum missing characters')
# p.add_argument('--missing_char_max', default=100, type=int, help='minimum missing characters')
# p.add_argument('--pred_guess', default=False, type=bool, help='predict guessed characters')
# p.add_argument('--log_dir', default=os.getcwd() + '/tf/log/', type=str, help='logging directory')
# p.add_argument('--loglevel', default='INFO', type=str, metavar='LEVEL',
#                help='Log level, will be overwritten by --debug. (DEBUG/INFO/WARN)')
FLAGS = p.parse_args()

''' This functionality will take all pedecerto XML files stored in ./pedecerto/xml_files and turn them into
pandas dataframes with the columns author, title, book, line, syllable, length. This information can be used
as the ground truth for the neural network models in the rest of this program. The converted files will be stored
in pickle format in ./pedecerto/pedecerto_df. Processed files will be moved to the processed subdirectory.
'''
if FLAGS.pedecerto_conversion:
    print('Converting pedecerto XML into pedecerto dataframes with scansion labels')
    from pedecerto.textparser import Pedecerto_parser
    pedecerto_parse = Pedecerto_parser(source = util.cf.get('Pedecerto', 'path_xml_files'),
                                       destination = util.cf.get('Pickle', 'path_pedecerto_df'))

if FLAGS.sequence_labels_conversion:
    print('Converting pedecerto dataframes into sequence labeling lists')
    from crf import CRF_sequence_labeling
    crf_parse = CRF_sequence_labeling()
    crf_parse.convert_pedecerto_dataframes_to_sequence_labeling_list(source = util.cf.get('Pickle', 'path_pedecerto_df'),
                                                                     destination = util.cf.get('Pickle', 'path_sequence_labels'))

exit(0)

########
# MAIN #
########
class Vector:
    # Wrapper class for vectors in a pandas dataframe 
    def __init__(self, v):
        self.v = v

# Parameters to run each step
run_preprocessor = False
run_pedecerto = False
run_model_generator = False
add_embeddings_to_df = False 
run_neural_network = False

''' Run the preprocessor on the given text if needed.
This reads the text, cleans it and returns a list of syllables for now
To achieve this, the pedecerto tool is used
'''
if run_preprocessor:
    print('Running preprocessor')
    preprocessor = Text_preprocessor(util.cf.get('Text', 'name'))
    util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'char_list'), preprocessor.character_list)

# Load the preprocessed text
character_list = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'char_list'))
if int(util.cf.get('Util', 'verbose')): print(character_list)

''' Now create a dataframe. Containing: syllable, length, vector.
'''
if run_pedecerto:
    print('Running pedecerto parser')
    parse = Pedecerto_parser(util.cf.get('Pedecerto', 'path_texts'))
    # This function created pickle files for all texts that are in the ./texts/ folder

pedecerto_df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'pedecerto_df'))
if util.cf.get('Util', 'verbose'): print(pedecerto_df)

# Run the model generator on the given list if needed
if run_model_generator:
    print('Running Word2Vec model generator')
    # Create a word2vec model from the provided character list
    word2vec_creator = Word_vector_creator(character_list, util.cf.getint('Word2Vec', 'vector_size'), util.cf.getint('Word2Vec', 'window_size'))
    util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'word2vec_model'), word2vec_creator.model)

# Load the saved/created model
word2vec_model = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'word2vec_model'))

# Add the embeddings created by word2vec to the dataframe
if add_embeddings_to_df:
    print('Adding embeddings to the dataframe')

    df = pedecerto_df
    df['vector'] = df['syllable']

    # Add syllable vectors to the dataframe using the word2vec model
    unique_syllables = set(df['syllable'].tolist())

    for syllable in Bar('Processing').iter(unique_syllables):
        try:
            vector = word2vec_model.wv[syllable]
            # Pump (wrapped) vector to the applicable positions
            df['vector'] = np.where(df['vector'] == syllable, Vector(vector), df['vector'])
        except:
            IndexError('Syllable has no embedding yet.')

    util.Pickle_write(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'embedding_df'), df)

# Provide the neural network with the dataframe
if run_neural_network:
    print('Running the neural network generation')

    df = util.Pickle_read(util.cf.get('Pickle', 'path'), util.cf.get('Pickle', 'embedding_df'))
    if util.cf.get('Util', 'verbose'): print(df)

    nn = Neural_network_handler(df)




