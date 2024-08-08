"""
File: moco_divshift_dataset.py
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
#import cv2
#from tfrecord.torch.dataset import MultiTFRecordDataset

# ----------------- Data augmentation parameters ----------------- #


RS_IMG_MEANS = [0.4859063923358917, 0.4978790581226349, 0.4434129297733307, 0.5152429342269897]
RS_IMG_STDS = [0.16339322924613953, 0.1428724229335785, 0.11196965724229813, 0.15414316952228546]


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



class NMVPretrainDataset(Dataset):
    def __init__(self, base_dir, aug_plus=True, data_split="!supervised"):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - base_dir (string): directory holding data
        - data_split (string): column title for split (with ! to reverse), '' for no split
        """
        self.base_dir = base_dir
        states = ['alaska', 'arizona', 'baja_california', 'baja_california_sur', 'british_columbia', 'california', 'nevada', 'oregon', 'sonora', 'washington', 'yukon']
        df = pd.concat({state_name : pd.read_csv(f'{base_dir}/{state_name}/observations_postGL.csv') for state_name in states})
        if (len(data_split) > 0 and data_split[0] == '!'):
            self.df = df[(~df[data_split[1:]]) & (df['download_success'] == 'yes')]
        elif len(data_split) > 0:
            self.df = df[df[data_split] & df['download_success'] == 'yes']
        else:
            self.df = df[df['download_success'] == 'yes']

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

        # ground level image      
        gl_img = self.load_image(idx)
        q, k = gl_img
        q = q.float()
        k = k.float()
        return [q, k]   

    
class CustomRotation(object):
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, img):
        angle = np.random.choice(self.angles)
        return transforms.functional.rotate(img, angle.item())
    

    
    
# ----------------- AA functions ----------------- #

"""        
def decode_gl_image(features, resize=(256, 256)):
    # get BGR image from bytes
    gl = cv2.imdecode(np.asarray(bytearray(features['streetlevel/encoded']), dtype="uint8"),cv2.IMREAD_COLOR)
    # convert BGR to std RGB
    gl = cv2.cvtColor(gl, cv2.COLOR_BGR2RGB)
    gl = Image.fromarray(gl)
    # gl = torch.tensor(gl)
    # gl = transforms.functional.convert_image_dtype(gl)
    # gl = transforms.functional.to_dtype(gl, dtype=torch.float, scale=True)
    # gl = gl.permute((2,0,1))
    # gl = transforms.functional.resize(gl, resize, antialias=True)
    features['streetlevel/encoded'] = gl
    return features
"""
def check_index(base_dir, splits):
    
    for file in splits.keys():
        if not os.path.exists(f"{base_dir}tfrecords/{file}.index"):
            index_cities(base_dir)
            raise ValueError(f"index file missing for {file}! Please run {base_dir}index_cities.sh to generate index file")

# generate tfrecord indexing command for a given city
def index_city(cdir):
    files = glob(f"{cdir}/*25")
    return [f"python3 -m tfrecord.tools.tfrecord2idx {f} {f}.index\n" for f in files]

# generate bash script to generate tfrecord indices
def index_cities(base_dir, cities:list=None):
    cdirs = glob(f"{base_dir}tfrecords/*/") if cities is None else [f"{base_dir}tfrecords/{c}" for c in cities]
    with open(f"{base_dir}/index_cities.sh", 'w') as f:
        [f.write(line) for cdir in cdirs for line in tqdm(index_city(cdir), desc='generating index files')]

# convert name from csv format to tfrecords format
def update_name(name):
    return re.sub(r"(\w)([A-Z])", r"\1_\2", name).lower()

# get the relative number of images per-city to 
# ensure uniformly sampling from whole dataset
def get_probabilities(base_dir, split, cities : list=None):

    files = glob(f"{base_dir}tree_locations/*Trees_{split}.csv") if cities is None else [f"{base_dir}tree_locations/{c}Trees_{split}.csv" for c in cities]
    csvs = {c.split('/')[-1].split('Trees')[0] : pd.read_csv(c) for c in tqdm(files, desc='generating dataset probabilities')}
    total = sum([len(c) for c in csvs.values()])
    return total, {update_name(n) : len(c)/total for (n,c) in csvs.items()}
    
def generate_splits(base_dir, split, probabilities, cities : list=None):
    
    cdirs = glob(f"{base_dir}tfrecords/*/") if cities is None else [f"{base_dir}tfrecords/{c}/" for c in cities]
    splits = {}
    
    for cdir in cdirs:
        partitions = glob(f"{cdir}{split}*25")
        cprob = probabilities[cdir.split('/')[-2]]
        for p in partitions:
            splits[p.split('tfrecords/')[1]] = cprob / len(partitions)
    return splits

def aa_collate_pretrain_gl(batch):
    # ulgy, but says that we're always using mocov2 augmentations which we are
    aug_plus = True
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                # chnanging to values that worked for swav
                [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
        ])
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
        ])
 
 
    gl1, gl2 = zip(*[(augmentation(b['streetlevel/encoded']), augmentation(b['streetlevel/encoded'])) for b in batch])
    gl1, gl2 = torch.stack(gl1), torch.stack(gl2)
    return gl1, gl2


def aa_collate_pretrain_rs(batch):
    augmentation = transforms.Compose([
        transforms.RandomCrop((100, 100)),
        transforms.RandomHorizontalFlip(.5),
        transforms.RandomVerticalFlip(.5),
        CustomRotation(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    ])

    rs1, rs2 = zip(*[(augmentation(b['aerial/encoded']), augmentation(b['aerial/encoded'])) for b in batch])
    rs1, rs2 = torch.stack(rs1), torch.stack(rs2)
    return rs1, rs2


def aa_collate_finetune_both(batch):
    gl, rs, label = zip(*[(b['streetlevel/encoded'], b['aerial/encoded'], b['tree/genus/label']) for b in batch])
    gl, rs, label = torch.stack(gl), torch.stack(rs), torch.tensor(np.stack(label))
    return (gl, rs), label

def aa_collate_finetune_gl(batch):
    gl, label = zip(*[(b['streetlevel/encoded'], b['tree/genus/label']) for b in batch])
    gl, label = torch.stack(gl),torch.tensor(np.stack(label))
    return gl, label

"""
def config_aa(base_dir, view, split='train', cities=None, resize=RESIZE, infinite=False):
    global RESIZE
    if resize != RESIZE:
        RESIZE = resize
        print(f'resize is now {RESIZE}')
    tfrecord_pattern = base_dir + 'tfrecords/{}'
    index_pattern = base_dir + 'tfrecords/{}.index'
    nrecords, probs = get_probabilities(base_dir, split=split, cities=cities)
    if cities is not None:
        cities = [update_name(c) for c in cities]
    splits = generate_splits(base_dir, split=split, probabilities=probs, cities=cities)
    check_index(base_dir, splits)
    if view == 'ground_level':
        return nrecords, MultiTFRecordDataset(tfrecord_pattern, 
                                       index_pattern, 
                                       splits, 
                                       AA_DESCRIPTION,
                                       transform=decode_gl_image, 
                                       infinite=infinite, 
                                       shuffle_queue_size=2)
    
"""    