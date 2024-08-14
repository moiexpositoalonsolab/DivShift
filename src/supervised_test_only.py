"""
File: supervised_train.py
Authors: Elena Sierra & Lauren Gillespie
------------------
Benchmark different domain shifts using supervised ResNets
"""

import supervised_dataset
import supervised_utils as utils

# Torch packages / functions
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms

# Miscellaneous packages
import os
import time
import json
import socket
import random
import argparse
from tqdm import tqdm, trange
from datetime import datetime
import pandas as pd
import dask.dataframe as dd
from types import SimpleNamespace
import pdb


# ----------------- Training ----------------- #
def inference(args):
    
    # dataset
    print('setting up dataset')
    label_dict = {}
    if args.dataset == "DivShift":
        # Standard transform for ImageNet
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.Normalize(mean=[0.485,
                                                                   0.456,
                                                                   0.406],
                                                             std=[0.229,
                                                                  0.224,
                                                                  0.225]),])
        # Get training data
        ddf = pd.read_csv(f'{args.data_dir}/splits_lauren.csv')
        if (args.train_split in ddf.columns):
            train_df = ddf[(ddf['supervised'] == True) & (ddf[args.train_split] == 'train')]
        elif (args.train_split == '2019-2021'):
            train_df = ddf[(ddf['supervised'] == True) &
               ((pd.to_datetime(ddf['date']).dt.year == 2019) | (pd.to_datetime(ddf['date']).dt.year == 2020) | (pd.to_datetime(ddf['date']).dt.year == 2021))]
        else:
            raise ValueError('Please select a valid train_split')

        # associate class with index
        print(f"train df is this size: {train_df.shape} with {len(train_df['name'].unique())} labels")
        label_dict = {spec: i for i, spec in enumerate(sorted(train_df[args.to_classify].unique().tolist()))}
        print(f"label dict is {len(label_dict)} with {min(list(label_dict.values()))} min label name and {max(list(label_dict.values()))} max label name")
 
        print('setting up test dataset')
        # Get test data
        if (args.test_split in ddf.columns):
            if args.use_entire_split:
                test_df = ddf[(ddf['supervised'] == True) & ((ddf[args.test_split] == 'test') | (ddf[args.test_split] == 'train'))]
            else:
                test_df = ddf[(ddf['supervised'] == True) & (ddf[args.test_split] == 'test')]
        elif (args.test_split == '2022'):
            test_df = ddf[(ddf['supervised'] == True) & (ddf['download_success'] == 'yes') & (pd.to_datetime(ddf['date']).dt.year == 2022)]
        elif (args.test_split == '2023'):
            test_df = ddf[(ddf['supervised'] == True) & (ddf['download_success'] == 'yes') & (pd.to_datetime(ddf['date']).dt.year == 2023)]
        else:
            raise ValueError('Please select a valid test_split')

        test_df = test_df.loc[test_df[args.to_classify].isin(label_dict)]

        print(f"test df is this size: {test_df.shape} with {len(test_df['name'].unique())} labels")

        test_image_dir = args.data_dir

        test_dset = supervised_dataset.LabelsDataset(test_df, test_image_dir,
                                                  label_dict, args.to_classify,
                                                  transform=transform,
                                                  target_transform=None)
        test_loader = DataLoader(test_dset, args.test_batch_size,
                                 shuffle=True, num_workers=args.processes)

    # device
    device = torch.device(f"cuda:{args.device}" if args.device >=0 else "cpu")
    print(f"Experiment running on device: {device}")

    # model

    # read hyperparameters of model
    metadata = f"{args.model_dir}/finetune_results/{args.exp_id}/{args.exp_id}_hyperparams.json"
    modelweights = f"{args.model_dir}/finetune_results/{args.exp_id}/{args.exp_id}_best_model.pth" 
    with open(metadata, 'r') as f:
        hyperparams = json.load(f)
        hyperparams = SimpleNamespace(**hyperparams)
    
    
    
    modeldata = torch.load(modelweights, map_location=torch.device('cpu'))
    bestepoch = modeldata['epoch'] + 1
    if (args.model == 'ResNet50'):
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, len(label_dict))
        model.load_state_dict(modeldata["model_state_dict"], strict=True)
    else:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(label_dict))
        model.load_state_dict(modeldata["model_state_dict"], strict=True)
    lastmodel = f"{args.model_dir}/finetune_results/{args.exp_id}/{args.exp_id}_epoch{hyperparams.num_epochs - 1}.pth"
    if not os.path.exists(lastmodel):
        raise ValueError(f"WARNING: {args.exp_id} has not finished training! Best epoch is {bestepoch} and total number of expeced epochs is {hyperparams.num_epochs }")
    
    model.to(device)
    print(f'starting test')
    
    model.eval()
    all_logits, all_labels = [], []
    test_loss = 0.0
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader), 
                                 desc=f'testing model from epoch {bestepoch}'):
            
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_logits.append(output.detach().cpu())
            all_labels.append(target.detach().cpu())
            
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

    all_logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    labels = labels.detach().cpu()
    probits = F.softmax(all_logits, dim=1)
    probits = probits.detach().cpu()
    top1, top5 = utils.obs_topK(labels, probits, K=5) 
    spectop1 = utils.species_topK(labels, probits, K=1)
    spectop5 = utils.species_topK(labels, probits, K=5)
    spectop30 = utils.species_topK(labels, probits, K=30)
    weighttop1 = utils.rarity_weighted_topK(labels, probits, K=1)
    weighttop5 = utils.rarity_weighted_topK(labels, probits, K=5)
    weighttop30 = utils.rarity_weighted_topK(labels, probits, K=30)

    fartop1 = utils.subset_topK(labels, probits, test_dset.farlabs, 1)
    fartop5 = utils.subset_topK(labels, probits, test_dset.farlabs, 5)
    fartop30 = utils.subset_topK(labels, probits, test_dset.farlabs, 30)

    cartop1 = utils.subset_topK(labels, probits, test_dset.carlabs, 1)
    cartop5 = utils.subset_topK(labels, probits, test_dset.carlabs, 5)
    cartop30 = utils.subset_topK(labels, probits, test_dset.carlabs, 30)

    rartop1 = utils.subset_topK(labels, probits, test_dset.rarlabs, 1)
    rartop5 = utils.subset_topK(labels, probits, test_dset.rarlabs, 5)
    rartop30 = utils.subset_topK(labels, probits, test_dset.rarlabs, 30)


    # ecoregion
    eco_results1 = {}
    eco_results5 = {}
    for ecoregion, idxs in test_dset.l2_ecoregion.items():
        sublabels = labels[idxs]
        subpreds = probits[idxs]
        e1, e5 = utils.obs_topK(sublabels, subpreds, K=5)
        eco_results1[f"{ecoregion}_top_1"] = e1
        eco_results5[f"{ecoregion}_top_5"] = e5
    etop1 = np.mean(eco_results1.values())
    etop5 = np.mean(eco_results5.values())



    # land use category
    luc_results1 = {}
    luc_results5 = {}
    for ecoregion, idxs in test_dset.land_use.items():
        sublabels = labels[idxs]
        subpreds = probits[idxs]
        e1, e5 = utils.obs_topK(sublabels, subpreds, K=5)
        luc_results1[f"luc_{ecoregion}_top_1"] = e1
        luc_results5[f"luc_{ecoregion}_top_5"] = e5
    luctop1 = np.mean(luc_results1.values())
    luctop5 = np.mean(luc_results5.values())

    # TODO: update extra tidbits to save
    results = {
            'best_epoch' : [bestepoch],
            'model' : [hyperparams.model],
            'exp_id' : [args.exp_id],
            'test_split' : [args.test_split],
            'use_all_test' : [args.use_entire_split],
            'train_split' : [hyperparams.train_split],
            'train_type' : [hyperparams.train_type],
            'learning_rate' : [hyperparams.learning_rate],
            'batch_size' : [hyperparams.batch_size],
            'optimizer' : [hyperparams.optimizer],
            'date' : [datetime.datetime.now()],
            'obs_top_1' : [top1],
            'obs_top_5' : [top5],
            'obs_top_30' : [top30],
            'spec_top_1' : [spectop1],
            'spec_top_5' : [spectop5],
            'spec_top_30' : [spectop30],
            'weighted_top_1' : [weighttop1],
            'weighted_top_5' : [weighttop5],
            'weighted_top_30' : [weighttop30],
            'far_top_1' : [fartop1],
            'far_top_5' : [fartop5],
            'far_top_30' : [fartop30],
            'car_top_1' : [cartop1],
            'car_top_5' : [cartop5],
            'car_top_30' : [cartop30],
            'rar_top_1' : [rartop1],
            'rar_top_5' : [rartop5],
            'rar_top_30' : [rartop30],
            'eco_top_1' : [etop1],
            'eco_top_5' : [etop5],
            'luc_top_1' : [luctop1],
            'luc_top_5' : [luctop5],
            'test_loss' : [test_loss],
           }
    res = pd.DataFrame({**results, **eco_results1, **eco_results5, **luc_results1, **luc_results5})
    # save accs to group csv
    total_csv = f"{args.data_dir}/inference/{args.dataset}_{args.train_split}_overall.csv"
    print('saving overall results')
    if not os.path.exists(total_csv):
        res.to_csv(total_csv, index=False)
    else:
        tot_csv = pd.read_csv(total_csv)
        res = pd.concat([tot_csv, res])
        res.to_csv(total_csv, index=False)




# ----------------- Runner ----------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify cli arguments.", allow_abbrev=True)

    parser.add_argument('--device', type=int, help='what device number to use (-1 for cpu)', default=-1)
    parser.add_argument("--data_dir", type=str, help="Location of directory where train/test/val split are saved to.", required=True) 
    parser.add_argument("--model_dir", type=str, help="Location of directory where model weights are saved to.", required=True) 
    parser.add_argument('--dataset', type=str, help='DivShift', default='DivShift')
    parser.add_argument('--model', type=str, help='which model', choices=['ResNet18', 'ResNet50'], default='ResNet18')
    parser.add_argument("--exp_id", type=str, help="Experiment name of trained model.", required=True)
    parser.add_argument("--test_batch_size", type=int, help="Examples per batch", default=1000)
    parser.add_argument('--processes', type=int, help='Number of workers for dataloader.', default=0)
    parser.add_argument('--train_split', type=str, help="which split the saved weights were trained on", required=True)
    parser.add_argument('--test_split', type=str, help="which split to test on", required=True)
    parser.add_argument('--use_entire_split', action='store_true', help='for splits with few observations, opt to use the entire split')
    parser.add_argument('--to_classify', type=str, help="which column to classify", default='name')
    parser.add_argument('--testing', action='store_true', help='dont log the run to tensorboard')
    
    args = parser.parse_args()
    inference(args)
