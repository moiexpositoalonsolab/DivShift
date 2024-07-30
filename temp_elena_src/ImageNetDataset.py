"""
File: ImageNetDataset.py
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
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # determine image path
        photo_id_train = self.img_labels.iloc[idx]['photo_id']
        folder_num_train = photo_id_train // 1000000
        state = self.img_labels.iloc[idx]['state_name']
        img_path = self.img_dir + "/" + state + "/" + str(folder_num_train) + "/" + str(photo_id_train) + ".png"

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
        
        return tensor_image, label