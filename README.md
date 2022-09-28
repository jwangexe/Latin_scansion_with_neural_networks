# Scanning Latin poetry with machine learning
![alt text](https://github.com/Ycreak/Latin_scansion_with_neural_networks/blob/main/banner.png "Ancient Tetris professional")

## About
Welcome to the project for scanning Latin poetry with machine learning. The goal of this project is to train machine learning models on dactylic meters and test the generalisability to other meters such as the anapest and glyconeus. 

The project consists of three parts: 

1. Creating data sets for machine learning tools
2. Training and testing an CRF on Latin poetry
3. Training and testing an LSTM on Latin poetry

For more information about this project, see the [LUCAS website](https://www.universiteitleiden.nl/en/humanities/centre-for-digital-humanities/projects/small-grant-projects#developing-an-intelligent-metrical-analysis-tool-for-latin-poetry) and my [thesis](https://github.com/Ycreak/Latin_scansion_with_neural_networks/blob/main/thesis.pdf) about the project.

### Table of Contents  
+ [Requirements](#Requirements)  
+ [Dataset creation](#Dataset)  
+ [Conditional Random Fields](#CRF)  
+ [Long short-term memory](#LSTM)  

<a name="Requirements"/>

## Requirements
The programs are written entirely in Python and need the following packages:

1. pandas
2. bs4
3. lxml
4. keras
5. tensorflow
6. sklearn_crfsuite
7. cltk
8. matplotlib
9. seaborn
10. configparser
11. progress

All dependencies are also listed in requirements.txt. In your Python environment, you can install all dependencies via pip using the following command:

```console 
pip install -r requirements.txt
```

As is the main problem with machine learning, packages are out of date the moment you've installed them. It worked on my machine :D

<a name="Dataset"/>

## Creating a data set
Before machine learning can be trained and tested on Latin poetry, we need to create a data set that is machine readable. For this, we used a syllable-label list. Here, every syllable in a line of poetry is accompanied by its label, being either _long_, _short_, or _elision_. We represent every line with a list, with tuples for every syllable-label combination. To illustrate, the first three syllables of Vergil's _Aeneid_ would look as follows:

```python
[(ar, long), (ma, short), (vi, short)]
```

An entire text of combinations of texts would then be represented of a list of lists, with each nested list being a line of poetry. To create such a list, two methods can be employed, as seen below.

### Using Pedecerto
The [Pedecerto](https://www.pedecerto.eu/public/) project uses a rule-based approach to scan dactylic meter. We use their scansions to create our dataset. The scansions are made available in the form of XML files on [their website](https://www.pedecerto.eu/public/pagine/autori). Now do the following:

1. Download the XML files of which you want to create your dataset. 
2. Place the downloaded XML files in the **pedecerto/xml_files/** folder.
3. Run the **main.py** program with the following parameters:

```console 
python3 main.py --pedecerto_conversion
```

  This will create pickle files with syllable-label lists in the **pedecerto/xml_files/**, moving any processed XML files to the **processed** folder. 

4. (Optional) You can combine the syllable label list of multiple texts and authors by putting the generated pickle files in the **combine** folder and running the following command:

```console 
python3 main.py --combine_author_files
```

Optionally, you can provide a name for the output file using the following command:

```console 
python3 main.py --combine_author_files --outputname <your_file_name>
```

This will store the created file in the **pickle/sequence_labels** folder. 

**IMPORTANT** Make sure that all your pickled syllable-label lists are stored in the **pickle/sequence_labels** folder, as this is the folder the machine learning tools will use when searching for datasets.

_Note: the tool that creates the syllable-label lists will only process hexameters and pentameters, as these are the focus of the Pedecerto project. Any corrupt lines will be discarded, as well as lines containing words that cannot be syllabified by the syllabifier provided by BNagy and his [MQDQ-parser](https://github.com/bnagy/mqdq-parser) (found under pedecerto/syllabifier.py and pedecerto/rhyme.py)._

### Using Anceps
The [Anceps]([https://www.pedecerto.eu/public/](https://github.com/Dargones/anceps)) project uses a constraint-based approach to scan iambic trimeter. We also provide tools to convert these scansions into syllable-label lists. However, these lists will contain the extra label _anceps_ as Anceps does not resolve these labels automatically.

1. Create scansion files using the Anceps tools, or download complete scansions from the [Senecan Trimeter and Humanist Tragedy repository](https://github.com/QuantitativeCriticismLab/AJP-2022-Senecan-Trimeter).
2. Put the JSON files in the **anceps/full_scansions** folder and run the following command:

```console 
python3 trimeter.py --create_syllable_file
```
This will create a pickled syllable_label file of each JSON in the **pickle/sequence_labels** folder. To combine these files, the combining code described in the previous section can be used (move pickled files to the combine folder and run the pickle combining command).

<a name="CRF"/>

## Running the CRF Network
There are two options to run the CRF network. 

1. The first one uses k-fold cross validation and runs on a single pickled file. To invoke this functionality, run the following code:

```console 
python3 main.py --crf --kfold --train <your_file_name.pickle>
```

2. The second option is to train on one pickled file and test on another:

```console 
python3 main.py --crf --custom_train_test --train <your_file_name.pickle> --test <your_file_name.pickle>
```

The scores will be printed in the form of a metrics report.

<a name="LSTM"/>

## Running the LSTM Network
There are, like with the CRF model, two options to run the LSTM network. 

1. The first one uses k-fold cross validation and runs on a single pickled file. To invoke this functionality, run the following code:

```console 
python3 main.py --lstm --kfold --train <your_file_name.pickle> --create_model
```

2. The second option is to train on one pickled file and test on another:

```console 
python3 main.py --lstm --custom_train_test --train <your_file_name.pickle> --test <your_file_name.pickle> --create_model
```

_Note: to train a new model, add the --create_model parameter. Optionally, one can save the model to disk with the --save_model parameter. To load a trained model, simply remove the --create_model parameter, which load the saved_model from disk based on the file name of the training pickle._

Additionally, it is possible to specify the number of epochs using the **--epoch** parameter. 

As with the CRF, the scores will be printed in the form of a metrics report.
