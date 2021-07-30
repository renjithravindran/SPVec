# Syntagmatic Word Embeddings
This repository contains a python implementation of the syntagmatic Word Embedding model of Renjith P Ravindran, Akshay Badola and K Narayana Murthy as described in their paper [Syntagmatic Word Embeddings for Unsupervised Learning of Selectional Preferences](http://aclanthology.org/2021.repl4nlp-1.22/).
Unlike the embedding models such as Word2vec and GloVe, that capture word similarity (paradigmatic), syntagmatic embeddings capture word associations (syntagmatic relations).
Therefore, it can be used to measure the degree of selectional preference between two words.
Selectional preferences of words tell how likely a two words are to form a syntactic relation.
For eg. `black cat' is more likely than `blue cat', or that `eat dinner' is more likely than `eat tree'
Selectional preferences are usually learned from a parsed corpus, but syntagmatic embeddings can do pretty well simply with a plain corpus.


## Make 
```
python3 spvec.py <path-to-corpus>
```
