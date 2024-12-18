"""
File: crisp_datasetransforms.py
------------------
Defines the PyTorch dataset classes.
"""

# torch packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# misc packages
import os 
import json 
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from glob import glob
import re
from PIL import ImageFilter



IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]


global RESIZE
RESIZE  = (224,224)




class DivShiftPretrainDataset(Dataset):
    def __init__(self, base_dir, exclude_supervised=True):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - base_dir (string): directory holding data
        - data_split (string): train, validation, or test
        """
        self.base_dir = base_dir
        df = pd.read_csv(f"{base_dir}allobs_postGL_presplit.csv") # TODO: 
        if exclude_supervised:
            self.df = df[~df.supervised]
            self.df.reset_index(inplace=True)
        else:
            self.df = df
    
        
        self.imagenet_means = IMAGENET_MEANS
        self.imagenet_stds = IMAGENET_STDS
        self.gl_normalize = transforms.Normalize(self.imagenet_means, self.imagenet_stds)

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic, 224 default input size for MAE
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.gl_normalize,
        ])
        

    def load_image(self, idx):

        
        x = Image.open(self.img_loc).convert('RGB') # new dataet
        image_array = np.array(x) 

        # remove transparency dimension if it exists 
        if image_array.shape[-1] == 4 and len(image_array.shape) == 3:
            image_array = image_array[:, :, :3]
        elif image_array.shape[0] == 4 and len(image_array.shape) == 3:
            image_array = image_array[:3, :, :]

        # if image has only 2 channels, add a third channel
        if len(image_array.shape) == 3 and image_array.shape[-1] == 2:
            image_array = np.concatenate([np.expand_dims(image_array[:, :, -1], axis=-1), image_array], axis=-1)
        elif len(image_array.shape) == 3 and image_array.shape[0] == 2:
            image_array = np.concatenate([np.expand_dims(image_array[-1, :, :], axis=0), image_array], axis=0)


        # if image is grayscale, add a channel dimension
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=0)

        assert len(image_array.shape) == 3
        x = Image.fromarray(image_array)

        x = self.augmentation(x)
        return x
        

    
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.df)

        
    def __getitem__(self, idx):
        """
        Return the image at the given index. 
        """

        state = self.df.iloc[idx]['state_name']
        photo_id = str(self.df.iloc[idx]['photo_id'])
        folder_num = photo_id[:3]
        
        self.img_loc = f"{self.base_dir}/{state}/{folder_num}/{photo_id}.png"
        
        img = self.load_image(idx)
        return img 

    
