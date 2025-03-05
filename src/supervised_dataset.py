"""
File: supervised_dataset.py
Authors: Elena Sierra & Lauren Gillespie
------------------
Defines the PyTorch dataset classes
"""

# torch packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# misc packages
import os 
import json 
import scipy
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# ----------------- Dataset classes ----------------- #


# calculate jensen-shannon distance
def calculate_jsd(df, train_partition, test_partition, to_classify, b_partition=None, use_entire_split=False):

    if (b_partition is not None) & use_entire_split:
        raise ValueError("Warning! Trained on PAtrain and PBtrain but are testing on PBtest AND PBtrain!")
    # restrict to only observations used for training
    df = df[df.supervised]
    # P_a_train
    p_a_train = df[df[train_partition] == 'train']
    # include B partition if necessary
    if b_partition is not None:
        p_b_train = df[df[b_partition] == 'train']
        p_a_train = pd.concat([p_a_train, p_b_train])
        
    # P_b_test
    if use_entire_split: # for when certain splits are very small
        p_b_test = df[(df[test_partition] == 'test') | (df[test_partition] == 'train')]
    else:
        p_b_test = df[df[test_partition] == 'test']
    # P_a_test
    p_a_test = df[df[train_partition] == 'test']
    
    # inline with model training, only consider labels present in the training set
    to_keep = sorted(p_a_train[to_classify].unique().tolist())

    # get the count of each label in each partition subset
    p_a_train_dict = p_a_train[to_classify].value_counts()
    p_b_test_dict = p_b_test[to_classify].value_counts()
    p_a_test_dict = p_a_test[to_classify].value_counts()
    # filter label counts per partition subset to only 
    # consider those present in the training set
    # and ensure in same order so distance calculation lines up
    p_a_train_dist = [p_a_train_dict[spec] for spec in to_keep]
    # if a label isn't present in the test set, impute 0 observations for that species
    p_b_test_dist =  [p_b_test_dict[spec]  if spec in p_b_test_dict else 0 for spec in to_keep]
    p_a_test_dist =  [p_a_test_dict[spec]  if spec in p_a_test_dict else 0 for spec in to_keep]

    # base 2 to ensure 0-1 range
    patr_pbte_dist = scipy.spatial.distance.jensenshannon(p_a_train_dist, p_b_test_dist, base=2) 
    patr_pate_dist = scipy.spatial.distance.jensenshannon(p_a_train_dist, p_a_test_dist, base=2) 

    return patr_pbte_dist, patr_pate_dist
    
    

def randomize_train_test(df, partition, generator):
    # only pick from eligible observations
    obs = df[(~df[partition].isna()) & (df[partition] != 'not_eligible')]
    
    chosen = generator.choice(obs.index, math.floor(len(obs)*.2), replace=False)
    df.loc[chosen, partition] = 'test'
    notc = obs.index.difference(chosen)
    df.loc[notc, partition] = 'train'
    
    
    return df


def randomize_taxonomic_train_test(df, generator):
    
    # hyperparameters from the supplemental
    min_obs = 25
    max_obs = 300

    balanced_train = []
    all_train = []
    test = []
    for spec, dff in tqdm(df.groupby('name'), total=len(df.groupby('name'))):
        testidxs = generator.choice(dff.index, math.floor(len(dff)*.2), replace=False)
        remaining = dff.index.difference(testidxs).values
        btrain = generator.choice(remaining, max_obs, replace=False) if len(remaining) > max_obs else remaining
        balanced_train += btrain.tolist()
        all_train += remaining.tolist()
        test += testidxs.tolist()

    df['taxonomic_balanced'] = 'not_eligible'
    df['taxonomic_unbalanced'] = 'not_eligible'

    df.loc[balanced_train, 'taxonomic_balanced'] = 'train'
    df.loc[test, 'taxonomic_balanced'] = 'test'
    df.loc[all_train, 'taxonomic_unbalanced'] = 'train'
    df.loc[test, 'taxonomic_unbalanced'] = 'test'
    return df


# class providing tensor of image and its label for the DivShift dataset
class LabelsDataset(Dataset):
    def __init__(self, data_frame, img_dir, label_dict, to_classify, transform, target_transform=None):
        self.img_labels = data_frame
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_dict = label_dict
        self.to_classify = to_classify

        # load label map 
        meta_path = f"{img_dir}/divshift_nawc_far_car_rar.json"
        with open(meta_path, 'r') as file:
            metadata = json.load(file)
        # set up groupings for partition accuracy rankings
        self.far = metadata['far']
        self.car = metadata['car']
        self.rar = metadata['rar']

        self.farlabs = [self.label_dict[n] for n in self.far if n in self.label_dict.keys()]
        self.carlabs = [self.label_dict[n] for n in self.car if n in self.label_dict.keys()]
        self.rarlabs = [self.label_dict[n] for n in self.rar if n in self.label_dict.keys()]
        print(f"using {len(self.farlabs)} frequent species, {len(self.carlabs)} frequent species, and {len(self.rarlabs)} rare species")


        # ecoregions
        self.l2_ecoregion = {}
        df = data_frame.reset_index()
        for ecoregion, smalldf  in df.groupby('l2_ecoregion'):
            self.l2_ecoregion[ecoregion] = smalldf.index 
        # land use
        self.land_use = {}
        df = data_frame.reset_index()
        for ecoregion, smalldf  in df.groupby('land_use'):
            self.land_use[ecoregion] = smalldf.index 
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # determine image path
        photo_id = str(self.img_labels.iloc[idx]['photo_id'])
        folder = photo_id[:3]
        state = self.img_labels.iloc[idx]['state_name']
        img_path = self.img_dir + "/" + state + "/" + folder + "/" + photo_id + ".png"

        # open and transform image
        img_og = Image.open(img_path)
        tensor_img = transforms.ToTensor()(img_og)
        tensor_img = tensor_img[:3, : , : ]
        if tensor_img.shape[0] < 3:  # Check the number of channels
            tensor_img = tensor_img.repeat(3, 1, 1)
        if self.transform:
            tensor_img = self.transform(tensor_img)

        # collect label
        label = self.label_dict[self.img_labels.iloc[idx][self.to_classify]]
        if self.target_transform:
            label = self.target_transform(label)
        
        return tensor_img, label