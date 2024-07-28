import glob
import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np



"""
This is where you will define how you will get data to push through the neural network
all pytorch datasets need an __init__, __len__, and __getitem__ methods
"""
class Name_Of_Dataset(Dataset):
    # the init method actually creates the dataset object. Usually it will be
    # a pandas dataframe or a numpy array that stores your data. There are fancier
    # datasets you can create, but this is the most simple
    def __init__(self, DATA_PATH):
        ####### Class fields ##########        
        # ie:
#         self.path = DATA_PATH
#         self.data = pd.read_csv(self.path)
    
    # the len method returns how many observations are in the dataset.
    # depending on the dataset, this could get fancier (ie if you're using)
    # vcf data and you have one file for a diploid organism and your individual
    # data piece is one genome, then your length would be num_vcf * 2
    def __len__(self):
        return len(self.data)

    # the __getitem__ method describes how you return one observation from your
    # dataset. For example, if you're using genomes, then this method would go fetch
    # the vcf file off the system, load in the genome and do any data preprocessing you
    # need to get the data ready to feed into the neural network.
    # Note: the thing this method returns is the input to the neural network defined in
    # network.py called INPUT_DATA
    def __getitem__(self, idx):
        # idx is the index of the item from the dataset you want
        # for example:
        return self.data.iloc(idx)
