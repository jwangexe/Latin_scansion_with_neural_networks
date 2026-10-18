# TransScan: Scanning Latin dactylic verse with Transformers
![Banner](docs/banner.png "Ancient Tetris professional")

**▶ [Try the live demo](https://huggingface.co/spaces/jwangexe/latin-scansion)**: type a line of Latin verse and see every
syllable scanned as long, short or elided, by a Transformer, an LSTM or a CRF. It runs entirely in your browser.

## About this project

This repository extends **Luuk Nolden's Master's thesis project**,
[*Scanning Latin poetry with machine learning*](https://theses.liacs.nl/pdf/2021-2022-NoldenL.pdf) (Universiteit Leiden,
2022), and its original codebase, [Ycreak/Latin_scansion_with_neural_networks](https://github.com/Ycreak/Latin_scansion_with_neural_networks).
Nolden's work was part of the
[LUCAS small-grant project](https://www.universiteitleiden.nl/en/humanities/centre-for-digital-humanities/projects/small-grant-projects#developing-an-intelligent-metrical-analysis-tool-for-latin-poetry)
on an intelligent metrical analysis tool for Latin poetry.

**From Nolden's original project** (used here with only compatibility updates):
- the dataset pipeline that turns [Pedecerto](https://www.pedecerto.eu/) and [Anceps](https://github.com/Dargones/anceps) scansions into syllable-label lists (`lsnn/dataset_creation.py`, `lsnn/utilities.py`)
- the Conditional Random Field model (`lsnn/crf.py`)
- the LSTM model (`lsnn/lstm.py`)

**Added in this fork** by Yijun Wang, for the TransScan paper (*Scanning Latin dactylic verse using Transformers*):
- **a Transformer encoder for scansion** (`transformer.py`), trained in four sizes and compared with Nolden's LSTM
- an upgrade of the codebase to TensorFlow 2.x and Keras 3.x
- metric scripts for the LSTM and Transformer (`calc_metrics_lstm.py`, `calc_metrics_trans.py`)
- the [interactive demo](https://huggingface.co/spaces/jwangexe/latin-scansion) (`hf_space/`), along with trained models in `models/lstm/`

## Results

F1 per label on Pedecerto's dactylic hexameter and elegiac couplets (`HEX_ELE-all`, about 56,500 lines), from the
TransScan paper (Table 4):

| Model | Blocks | Heads | Emb. dim | FFN dim | Long | Short | Elision |
|---|---|---|---|---|---|---|---|
| LSTM (Nolden) | – | – | – | – | 0.9902 | 0.9861 | **0.9946** |
| Transformer Tiny | 1 | 4 | 32 | 32 | 0.9787 | 0.9703 | 0.8191 |
| Transformer Base | 2 | 4 | 48 | 48 | 0.9837 | 0.9828 | 0.8794 |
| Transformer Medium | 3 | 6 | 48 | 48 | 0.9929 | 0.9908 | 0.9462 |
| Transformer Large | 4 | 6 | 64 | 64 | **0.9937** | **0.9926** | 0.9380 |

All Transformers were trained for 25 epochs with Adam (learning rate 0.001) and a 0.2 validation split. The larger
models beat the LSTM on long and short syllables, which make up about 98% of the data, but every Transformer is weaker on
elided syllables. Nolden's CRF reaches a weighted F1 of 0.90.

### Table of contents
+ [Requirements](#requirements)
+ [Dataset creation](#dataset)
+ [Conditional Random Fields](#crf)
+ [Long short-term memory](#lstm)
+ [Transformer](#transformer)
+ [Demo](#demo)
+ [Credits](#credits)

<a name="requirements"></a>

## Requirements
The code is written entirely in Python. Install its dependencies with pip:

```console
pip install -r requirements.txt
```

<a name="dataset"></a>

## Creating a data set
*(Nolden's pipeline, unchanged.)*

Before a model can be trained on Latin poetry, the poetry needs to be in a machine-readable form: a syllable-label list.
Every syllable in a line is paired with its label: _long_, _short_ or _elision_. Each line is a list of
syllable-label tuples. For example, the first three syllables of Vergil's _Aeneid_ look like this:

```python
[(ar, long), (ma, short), (vi, short)]
```

A whole text is then a list of these lists, one per line of poetry. There are two ways to create one.

### Using Pedecerto
The [Pedecerto](https://www.pedecerto.eu/) project scans dactylic meter with a rule-based approach, and its scansions
are the source of the dataset. They are available as XML files on [the Pedecerto website](https://www.pedecerto.eu/pagine/autori).

1. Download the XML files for the texts you want in your dataset.
2. Put them in the **scansions/pedecerto/** folder.
3. Call **convert_pedecerto_xml_to_syllable_sequence_files** from the **dataset_creation** module:

```python
from lsnn import dataset_creation

dataset_creation.Pedecerto_parser().convert_pedecerto_xml_to_syllable_sequence_files(
    input_folder = conf.PEDECERTO_SCANSION_FOLDER,
    output_folder = conf.SEQUENCE_LABELS_FOLDER
)
```
This saves one pickle file of syllable-label lists per text in **scansions/sequence_labels/**.

4. (Optional) Combine the syllable-label lists of several texts:

```python
from lsnn import utilities as util

util.combine_sequence_label_lists(
    list_with_file_names = util.create_files_list(conf.SEQUENCE_LABELS_FOLDER, 'pickle'),
    output_name = 'combined.pickle',
    destination_path = conf.SEQUENCE_LABELS_FOLDER,
    add_extension = False
)
```

This combines every pickled list in the sequence_labels folder into **combined.pickle**. To combine only some files,
pass a list of names instead:

```python
from lsnn import utilities as util

util.combine_sequence_label_lists(
    list_with_file_names = ['HOR-arpo', 'ENN-anna'],
    output_name = 'combined.txt',
    destination_path = conf.SEQUENCE_LABELS_FOLDER,
    add_extension = True
)
```

**Important:** keep all pickled syllable-label lists in **scansions/sequence_labels/**. The models look for datasets
there, and they build their syllable vocabulary from every pickle in it. The paths can be changed in **lsnn/config.py**.

_Note: the parser processes only hexameters and pentameters, the focus of the Pedecerto project. Corrupt lines are
discarded, as are lines with words that BNagy's [MQDQ-parser](https://github.com/bnagy/mqdq-parser) syllabifier can't split._

### Using Anceps
The [Anceps](https://github.com/Dargones/anceps) project scans iambic trimeter with a constraint-based approach. Its
scansions can also be converted into syllable-label lists. Anceps doesn't resolve _anceps_ syllables, so these lists
contain the extra label _anceps_; tell the LSTM and CRF models about the additional label.

1. Create scansion files with the Anceps tools, or download complete scansions from the [Senecan Trimeter and Humanist Tragedy repository](https://github.com/QuantitativeCriticismLab/AJP-2022-Senecan-Trimeter).
2. Put the JSON files in the **scansions/anceps/** folder and run:

```python
from lsnn import dataset_creation

dataset_creation.Anceps_parser().convert_anceps_json_to_syllable_sequence_files(
    input_folder = conf.ANCEPS_SCANSION_FOLDER,
    output_folder = conf.SEQUENCE_LABELS_FOLDER
)
```

This creates a pickled syllable-label file for each JSON file. To combine them, use the code from the previous section.

<a name="crf"></a>

## Running the CRF
*(Nolden's model.)* There are two ways to run the CRF.

1. Train on one pickled file and test on another:

```python
latin_crf = Latin_CRF()

result = latin_crf.custom_prediction(
    predictor = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'ENN-anna.pickle'),
    predictee = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HOR-arpo.pickle')
)
print(result)
```

2. Run k-fold cross-validation on a single pickled file:

```python
result = latin_crf.kfold_model(
    sequence_labels = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'ENN-anna.pickle'),
    splits = 5
)
print(result)
```

<a name="lstm"></a>

## Running the LSTM
*(Nolden's model, updated for TensorFlow 2.x / Keras 3.x.)* First, create the Latin_LSTM class, which builds the
syllable and label dictionaries:

```python
lstm = Latin_LSTM(
    sequence_labels_folder = conf.SEQUENCE_LABELS_FOLDER,
    models_save_folder = conf.MODELS_SAVE_FOLDER,
    anceps_label = False,
)
```

Next, train a model or load a saved one. Models are saved in the `.keras` format:

```python
model, history = lstm.create_model(
    text = 'HEX_ELE-all.pickle',
    num_epochs = 25,
    save_model = True,
    model_name = 'lstm_HEX_ELE-all'
)

model = lstm.load_model(
    path = conf.MODELS_SAVE_FOLDER + 'lstm_HEX_ELE-all.keras'
)
```

Then use the model to predict the labels of a sequence-label file:

```python
test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
result = lstm.predict_given_set(test_set, model)
```

_Note: every syllable in the text you want to predict must be in the syllable dictionary, or a KeyError occurs. Put
the pickle file you want to predict in **scansions/sequence_labels/** before creating the Latin_LSTM class, since the
dictionary is built from that folder. The model folders in `models/lstm/` without a `.keras` extension are Nolden's
original TensorFlow SavedModels. They were built with different vocabularies and can't be loaded by Keras 3._

<a name="transformer"></a>

## Running the Transformer
*(Added in this fork.)* `transformer.py` provides `Latin_Transformer`, a Transformer encoder for sequence labelling.
It has the same interface as `Latin_LSTM`. Each syllable gets a token embedding plus a position embedding, passes
through a Transformer block (multi-head self-attention and a feed-forward network, each followed by layer
normalisation), and is then classified with a softmax:

```python
from transformer import Latin_Transformer

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

test_set = util.pickle_read(conf.SEQUENCE_LABELS_FOLDER, 'HEX_ELE-all.pickle')
result = transformer.predict_given_set(test_set, model)
```

`get_model(embed_dim, num_heads, ff_dim)` sets the model size. To train the four sizes from the paper, which stack 1–4
Transformer blocks, run `hf_space/build/train_transformers.py`:

```console
python hf_space/build/train_transformers.py medium   # tiny | base | medium | large
```

To score predictions per label (F1 for long, short and elision), save the result with `np.save` and run
`calc_metrics_trans.py`, or `calc_metrics_lstm.py` for the LSTM. `test.py` shows the full train, predict and plot workflow.

<a name="demo"></a>

## Demo
**[huggingface.co/spaces/jwangexe/latin-scansion](https://huggingface.co/spaces/jwangexe/latin-scansion)**

The demo offers the Transformer Tiny, Base and Medium models alongside Nolden's LSTM and CRF. It splits typed verse
into syllables, labels each syllable, and checks whether the line fits a dactylic hexameter or pentameter. It runs in
the browser with [Pyodide](https://pyodide.org) and uses this repository's own `lsnn/` and `transformer.py` code. The
trained networks run as NumPy re-implementations, which were checked against the original Keras and crfsuite models.
Its source is in [`hf_space/`](hf_space/); [`hf_space/README.md`](hf_space/README.md) covers how it works and how to
rebuild the models.

<a name="credits"></a>

## Credits
- **Luuk Nolden** ([Ycreak](https://github.com/Ycreak)): the original project, dataset pipeline, CRF and LSTM, from the
  Master's thesis [*Scanning Latin poetry with machine learning*](https://theses.liacs.nl/pdf/2021-2022-NoldenL.pdf) (Universiteit Leiden, 2022).
- **Yijun Wang** ([jwangexe](https://github.com/jwangexe)): the TransScan Transformer models, the TensorFlow 2 / Keras 3
  upgrade, and the demo.
- Scansion data: [Pedecerto](https://www.pedecerto.eu/) (Università di Udine and Università Ca' Foscari Venezia),
  licensed [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/), and [Anceps](https://github.com/Dargones/anceps).
- Syllabifier: BNagy's [mqdq-parser](https://github.com/bnagy/mqdq-parser), adapted from the [Classical Language Toolkit](https://github.com/cltk/cltk).
