"""
File: supervised_utils.py
Author: Elena Sierra and Lauren Gillespie
------------------
Utility functions for: DivShift: Exploring Domain-Specific Distribution Shift in Volunteer-Collected Biodiversity Datasets
"""
import os
import glob
import json
import torch
import math
import numpy as np
import enum
from tqdm import tqdm
from types import SimpleNamespace
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

## ---------- MAGIC NUMBERS ---------- ##

# Standard image size
# to use during training
IMG_SIZE = 150
# number of channels in
# NAIP imagery: RGB-Infrared
NAIP_CHANS = 4
IMAGENET_CHANS = 3

## ---------- Data manipulation ---------- ##

def get_num_classes(json_file_path):
    # takes in label map json path and returns num classes
    with open(json_file_path, 'r') as file:
        label_map = json.load(file)

    num_keys = len(label_map.keys())

    return num_keys


# empty function call
def pass_(input):
    return input

# https://stackoverflow.com/questions/2659900/slicing-a-list-into-n-nearly-equal-length-partitions
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def dict_key_2_index(df, key):
    return {
        k:v for k, v in
        zip(df[key].unique(), np.arange(len(df[key].unique())))
    }

# more clear version of uniform scaling
# https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
def scale(x, min_=None, max_=None, out_range=(-1,1)):

    if min_ == None and max_ == None:
        min_, max_ = np.min(x), np.max(x)
    return ((out_range[1]-out_range[0])*(x-min_))/(max_-min_)+out_range[0]


## ---------- Accuracy metrics ---------- ##

# calculates intersection of two tensors
# assumed y_t is already in pres/abs form
def pres_intersect(ob_t, y_t):
    # only where both have the same species keep
    sum_ = ob_t + y_t
    int_ = sum_ > 1
    return torch.sum(int_, dim=1)

# calculates union of two tensors
def pres_union(ob_t, y_t):
    sum_ = ob_t + y_t
    sum_ = sum_ > 0
    return torch.sum(sum_, dim=1)

def precision_per_obs(ob_t: torch.tensor, y_t: torch.tensor):
    ob_t = torch.as_tensor(ob_t)
    y_t = torch.as_tensor(y_t)
    # get intersection
    top = pres_intersect(ob_t, y_t).float()
    # get # predicted species
    bottom = torch.sum(ob_t, dim=1)
    # divide!
    ans = torch.div(top, bottom)
    # if nan (divide by 0) just set to 0.0
    ans[ans != ans] = 0
    return ans

def recall_per_obs(ob_t, y_t):
    ob_t = torch.as_tensor(ob_t)
    y_t = torch.as_tensor(y_t)
    # get intersection
    top = pres_intersect(ob_t, y_t).float()
    # get # observed species
    bottom = torch.sum(y_t, dim=1)
    ans = torch.div(top, bottom)
    # if nan (divide by 0) just set to 0.0
    # this relies on the assumption that all nans are 0-division
    ans[ans != ans] = 0
    return ans

def accuracy_per_obs(ob_t, y_t):
    ob_t = torch.as_tensor(ob_t)
    y_t = torch.as_tensor(y_t)
    # intersection
    top = pres_intersect(ob_t, y_t).float()
    # union
    bottom = pres_union(ob_t, y_t)
    ans = torch.div(top, bottom)
    # if nan (divide by 0) just set to 0.0
    # this relies on the assumption that all nans are 0-division
    ans[ans != ans] = 0
    return ans

def f1_per_obs(ob_t, y_t):
    pre = precision_per_obs(ob_t, y_t)
    rec = recall_per_obs(ob_t, y_t)
    ans =  2*(pre*rec)/(pre+rec)
    # if denom=0, F1 is 0
    ans[ans != ans] = 0.0
    return ans

def zero_one_accuracy(y_true, y_preds, threshold=0.5):
    assert y_preds.min() >= 0.0 and(y_preds.max() <= 1.0), 'predictions must be converted to probabilities!'
    y_obs = y_preds >= threshold
    n_correct = sum([y_obs[i,label] for (i,label) in enumerate(y_true)])
    return n_correct / len(y_true)

def obs_topK(ytrue, yobs, K=5):
    yobs = torch.as_tensor(yobs)
    ytrue = torch.as_tensor(ytrue)
    _, tk = torch.topk(yobs, K)
    true_1 = ytrue.unsqueeze(1).repeat(1,1)
    true_k = ytrue.unsqueeze(1).repeat(1,K)
    # don't forget to average
    top1_match = (tk == true_1)[:, 0].sum().item() / len(ytrue)
    topk_match = (tk == true_k).sum().item() / len(ytrue)

    return top1_match, topk_match

def species_topK(ytrue, yobs, K):
    nspecs = yobs.shape[1]
    yobs = torch.as_tensor(yobs)
    ytrue = torch.as_tensor(ytrue)

    _, tk = torch.topk(yobs, K)
    # get all unique species label and their indices
    unq = torch.unique(ytrue, sorted=False, return_inverse=True)
    # make a dict to store the results for each species
    specs = {v.item():[] for v in unq[0]}
    # go through each row and assign it to the corresponding
    # species using the reverse_index item from torch.unique
    for val, row in zip(unq[1], tk):
        specs[unq[0][val.item()].item()].append(row)
    sas = []
    for i in range(nspecs):
        # ignore not present species
        spec = specs.get(i)
        if spec is None:
            sas.append(np.nan)
            continue
        nspecs += 1
        # spoof ytrue for this species
        yt = torch.full((len(spec),K), i)
        # and calculate 'per-obs' accuracy
        sas.append((torch.stack(spec)== yt).sum().item()/len(spec))
    sas = np.array(sas)
    gsas = sas[~np.isnan(sas)]
    return (sum(gsas) / len(gsas))

def subset_topK(labels, probits, sublabels, K):
    fyt, fyo = labels.clone(), probits.clone()
    fyt = torch.tensor([(i, y) for i, y in zip(range(len(fyt)), fyt.tolist()) if y in sublabels])
    idxs, fyt = zip(*fyt)
    fyt = torch.stack(fyt)
    idxs = torch.stack(idxs)
    fyo = fyo[idxs, :]
    return species_topK(fyt, fyo, K=K)


def rarity_weighted_topK(ytrue, yobs, K):
    nspecs = yobs.shape[1]
    yobs = torch.as_tensor(yobs)
    ytrue = torch.as_tensor(ytrue)

    _, tk = torch.topk(yobs, K)
    # get all unique species label and their indices
    unq = torch.unique(ytrue, sorted=False, return_inverse=True)
    # make a dict to store the results for each species
    specs = {v.item():[] for v in unq[0]}
    # go through each row and assign it to the corresponding
    # species using the reverse_index item from torch.unique
    for val, row in zip(unq[1], tk):
        specs[unq[0][val.item()].item()].append(row)
    sas = []
    denom = []
    for i in range(nspecs):
        # ignore not present species
        spec = specs.get(i)
        if spec is None:
            sas.append(np.nan)
            denom.append(np.nan)
            continue
        nspecs += 1
        # spoof ytrue for this species
        yt = torch.full((len(spec),K), i)
        # and calculate 'per-obs' accuracy
        sas.append((torch.stack(spec)== yt).sum().item()/len(spec)**2)
        denom.append(1/len(spec))

    
    sas = np.array(sas)
    denom = np.array(denom)
    num = sas[~np.isnan(sas)]
    denom = denom[~np.isnan(denom)]
    return (num.sum() / denom.sum())

