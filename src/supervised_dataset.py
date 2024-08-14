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
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# ----------------- Dataset classes ----------------- #

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
        meta_path = f"{img_dir}/splits_lauren_far_car_rar.json"
        with open(meta_path, 'r') as file:
            metadata = json.load(file)
        # set up groupings for split accuracy rankings
        far = metadata['far']
        car = metadata['car']
        rar = metadata['rar']

        self.farlabs = [self.label_dict[n] for n in far if n in self.label_dict.keys()]
        self.carlabs = [self.label_dict[n] for n in car if n in self.label_dict.keys()]
        self.rarlabs = [self.label_dict[n] for n in rar if n in self.label_dict.keys()]
        print(f"using {len(self.farlabs)} frequent species, {len(self.carlabs)} frequent species, and {len(self.rarlabs)} rare species")


        # ecoregions
        self.l2_ecoregions = {}
        df = data_frame.reset_index()
        for ecoregion, smalldf  in df.groupby('l2_ecoregion'):
            self.l2_ecoregions[ecoregion] = smalldf.index 
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