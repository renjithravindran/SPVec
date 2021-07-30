# Syntagmatic Word Embeddings
This repository contains a python implementation of the Syntagmatic Word Embedding model of Renjith P Ravindran, Akshay Badola and K Narayana Murthy as described in their paper [Syntagmatic Word Embeddings for Unsupervised Learning of Selectional Preferences](http://aclanthology.org/2021.repl4nlp-1.22/).
Unlike the embedding models such as Word2vec and GloVe, that capture word similarity (paradigmatic relations), syntagmatic embeddings capture word associations (syntagmatic relations).
Therefore, it can be used to measure the degree of selectional preference between two words.
Selectional preferences of words tell how likely two words are to form a syntactic relation.
For eg. 'black cat' is more likely than 'blue cat', or that 'eat dinner' is more likely than 'eat tree'.
Selectional preferences are usually learned from syntactically related word pairs in a parsed corpus, but by reducing syntactic relations to directions, syntagmatic embeddings can do pretty well simply with a plain corpus.


## Make embeddings
```
$ python3 spvec.py <path-to-corpus> syn
```
note: requires a preprocessed (min counted) corpus file. Download sample corpus (used in paper) [here](https://drive.google.com/file/d/1fE5kSBHct3bnZE0_NOh3mxixZHdKmPGJ/view?usp=sharing)

## Test embeddings
```
>>> from spvec_eval import SP_Query
>>> spq = SP_Query(<path-to-embedding-file>)
>>> spq.get_associations('food')
```
note: Download sample embedding-file [here](https://drive.google.com/file/d/1CQ--9Shrls0kf6pdoza8dDBWNe1SLBd2/view?usp=sharing)
