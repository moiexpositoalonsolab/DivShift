"""
File: moco_divshift_dataset.py
------------------
Adapted from https://github.com/moiexpositoalonsolab/crisp-private/blob/main/src/moco/moco/moco_crisp_dataset.py
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


# ----------------- Data augmentation parameters ----------------- #


IMAGENET_MEANS = [0.485, 0.456, 0.406]
IMAGENET_STDS = [0.229, 0.224, 0.225]

global RESIZE
RESIZE  = (256,256)

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        # https://discuss.pytorch.org/t/pil-gaussianblur-vs-tfv-gaussian-blur/134928/2
        x  = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x 
# ----------------- Dataset classes ----------------- #



class DivShiftPretrainDataset(Dataset):
    def __init__(self, base_dir, aug_plus=True, exclude_supervised=True):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - base_dir (string): directory holding data
        - data_split (string): column title for split (with ! to reverse), '' for no split
        """
        self.base_dir = base_dir
        df = pd.read_csv(f"{base_dir}allobs_postGL_presplit.csv") # TODO: eventually make this the split csv
        print(f"{len(df)} observations before")
        if exclude_supervised:
            self.df = df[~df.supervised]
            self.df.reset_index(inplace=True)
            print(f"{len(self.df)} observations after")
        else:
            self.df = df

        self.imagenet_means = IMAGENET_MEANS
        self.imagenet_stds = IMAGENET_STDS
        self.gl_normalize = transforms.Normalize(self.imagenet_means, self.imagenet_stds)

        if aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    # chnanging to values that worked for swav
                    [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.8  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.gl_normalize,
            ])
        else:
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
            self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
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

        q = self.augmentation(x)
        k = self.augmentation(x)
        return [q, k]
        

    
        
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

    
        gl_img = self.load_image(idx)
        q, k = gl_img
        q = q.float()
        k = k.float()
        return [q, k]   
