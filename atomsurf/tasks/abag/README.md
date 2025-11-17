We follow the task presented in Geometric Epitope Prediction.

The aim is, given an antigen and an antibody, given as two separate proteins, to detect residues involved in the binding
on each protein.

The data is originally obtained from https://github.com/Marco-Peg/GEP/tree/master

In the data, we only use the systems csv:
- train.csv
- val_pecan_aligned.csv'
- test70.csv

Save those in a directory data/abag/ and run `python preprocess.py`.

You are then ready to train a model.