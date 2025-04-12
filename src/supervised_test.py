"""
File: supervised_test.py
Authors: Elena Sierra & Lauren Gillespie
------------------
Testing code for: DivShift: Exploring Domain-Specific Distribution Shift in Volunteer-Collected Biodiversity Datasets
"""
# DivShift functions
import supervised_dataset
import supervised_utils as utils

# Torch packages / functions
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

# Stats / data packages
import numpy as np
import pandas as pd
from datetime import datetime

# Miscellaneous packages
import os
import time
import json
import socket
import random
import argparse
from tqdm import tqdm, trange
from types import SimpleNamespace







# ----------------- Training ----------------- #
# in case where ``inference`` called inside of train script, 
# args and hyperparams will be the same
def inference(args, test_partition, read_hyperparams=False, use_entire_split=False, csv_id=None):

    if read_hyperparams:
        # read hyperparameters of model
        metadata = f"{args.model_dir}/divshift_models/{args.exp_id}/{args.exp_id}_hyperparams.json"
        modelweights = f"{args.model_dir}/divshift_models/{args.exp_id}/{args.exp_id}_best_model.pth" 
        with open(metadata, 'r') as f:
            hyperparams = json.load(f)
            hyperparams = SimpleNamespace(**hyperparams)
        assert args.train_partition == hyperparams.train_partition, f"Train partition out of alignment with hyperparameters! {args.train_partition} vs {hyperparams.train_partition}"
    else:
        hyperparams = args

    print('setting up dataset')
    if args.dataset == "DivShift":
        # Standard transform for ImageNet
        # using antialias=None to be consistent w/ defaults for
        # transforms.Resize(256) command  from torchvision < 0.17
        # that models were trained with and to remove the annoying
        # warning for users training w/ a newer version of torchvision
        transform = transforms.Compose([transforms.Resize(256, antialias=None), 
                                        transforms.CenterCrop(224),
                                        transforms.Normalize(mean=[0.485,
                                                                   0.456,
                                                                   0.406],
                                                             std=[0.229,
                                                                  0.224,
                                                                  0.225]),])
        # Get training data
        ddf = pd.read_csv(f'{args.data_dir}/divshift_nawc.csv',low_memory=False)
        # only using research-grade obs ID'd down to the species+ level
        ddf = ddf[ddf.supervised]
        # re-assign obs in each partition to train/test if randomizing
        if hyperparams.b_partition is None:
            print(f"train partition: {args.train_partition}; test partition: {test_partition}")
        else:
            print(f"train partition: {args.train_partition} + {hyperparamas.b_partition}; test partition: {test_partition}")
        if hyperparams.randomize_partition is not None:
            # set seed for train/test split reproducibility
            rand_gen = np.random.default_rng(seed=hyperparams.randomize_partition)
            # taxonomic bias is an edge case
            if args.train_partition in ['taxonomic_balanced', 'taxonomic_unbalanced']:
                ddf = supervised_dataset.randomize_taxonomic_train_test(ddf, rand_gen)
            # otherwise, shuffle bias partitions
            else:
                ddf = supervised_dataset.randomize_train_test(ddf, hyperparams.train_partition, rand_gen)
                # if using it, also randomize B partition
                if hyperparams.b_partition is not None:
                    ddf = supervised_dataset.randomize_train_test(ddf, hyperparams.b_partition, rand_gen)
                # if test bias partition wasn't included during training, randomize it too
                if test_partition != hyperparams.train_partition:
                    ddf = supervised_dataset.randomize_train_test(ddf, test_partition, rand_gen)
                
        # get JSD between Pa train, Pb test, and Pa test
        jsd_patr_pbte, jsd_patr_pate = supervised_dataset.calculate_jsd(ddf, args.train_partition, test_partition, args.to_classify, hyperparams.b_partition, use_entire_split)

        # split out train data to help filter test data appropriately
        train_df = ddf[ddf[args.train_partition] == 'train']
        # include B partition if relevant
        if hyperparams.b_partition is not None:
            addl_df = ddf[ddf[hyperparams.b_partition] == 'train']
            train_df = pd.concat([train_df, addl_df])
            
        # associate classes with index
        label_dict = {spec: i for i, spec in enumerate(sorted(train_df[args.to_classify].unique().tolist()))}
        print(f"train df is size: {train_df.shape} with {len(label_dict)} labels")
        # Get test dataset set up
        print('setting up test dataset')
        # if a bias partition is small, you can test on all observations from it
        if use_entire_split:
            test_df = ddf[(ddf[test_partition] == 'test') | (ddf[test_partition] == 'train')]
        else:
            # grab test observations from given test split
            test_df = ddf[ddf[test_partition] == 'test']
        # make sure to only keep observations for species seen during training
        test_df = test_df.loc[test_df[args.to_classify].isin(label_dict)]

        print(f"test df is size: {test_df.shape} with {len(test_df['name'].unique())} labels")

        test_image_dir = args.data_dir

        test_dset = supervised_dataset.LabelsDataset(test_df, test_image_dir,
                                                  label_dict, args.to_classify,
                                                  transform=transform,
                                                  target_transform=None)
        # don't drop the last batch when testing so results reproducible
        test_loader = DataLoader(test_dset, args.test_batch_size,
                                 shuffle=True, num_workers=args.processes)

    # set up pytorch so it can see your GPU
    device = torch.device(f"cuda:{args.device}" if args.device >=0 else "cpu")
    print(f"Experiment running on device: {device}")
    # grab trained model
    modeldata = torch.load(modelweights, map_location=torch.device('cpu'))
    # +1 since epoch is 0-based indexing
    bestepoch = modeldata['epoch'] + 1
    # set up model based on architecture etc
    if hyperparams.model == 'ResNet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, len(label_dict))
    elif hyperparams.model == 'ViT-Base':
        model = torchvision.models.vit_b_16()
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(label_dict))
    elif hyperparams.model == 'ViT-Large':
        model = torchvision.models.vit_l_16()
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(label_dict))
    else:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(label_dict))
    model.load_state_dict(modeldata["model_state_dict"], strict=True)
    # put model on the gpu
    model.to(device)    
    # make sure model in eval mode for reproducibility
    model.eval()    
    print(f"loaded model from {modeldata['epoch']}")
    
    lastmodel = f"{args.model_dir}/divshift_models/{args.exp_id}/{args.exp_id}_epoch{hyperparams.num_epochs - 1}.pth"
    if not os.path.exists(lastmodel):
        raise ValueError(f"WARNING: {args.exp_id} has not finished training! Best epoch is {bestepoch} and total number of expected epochs is {hyperparams.num_epochs }")
    
   # actual inference loop 
    print(f'starting test')
    all_logits, all_labels = [], []
    # no need to accumulate gradients w/ no backwards pass
    # saves on GPU memory uses
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader), 
                                 desc=f'testing model from epoch {bestepoch}'):
            # grab predictions and correct labels for all test set examples
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_logits.append(output.detach().cpu())
            all_labels.append(target.detach().cpu())

    # save accs to group csv
    if csv_id is not None:
        if hyperparams.b_partition is not None:
            save_path = f"{args.model_dir}/inference_results/{args.dataset}_{args.train_partition}_{hyperparams.b_partition}_train_{test_partition}_test_{args.csv_id}_overall_{socket.gethostname()}"

        else:
            save_path = f"{args.model_dir}/inference_results/{args.dataset}_{args.train_partition}_train_{test_partition}_test_{args.csv_id}_overall_{socket.gethostname()}"
    else:
        if hyperparams.b_partition is not None:
            save_path = f"{args.model_dir}/inference_results/{args.dataset}_{args.train_partition}_{hyperparams.b_partition}_train_{test_partition}_test_overall_{socket.gethostname()}"
        else:
            save_path = f"{args.model_dir}/inference_results/{args.dataset}_{args.train_partition}_train_{test_partition}_test_overall_{socket.gethostname()}"

    # save predictions per-image if requested
    if args.save_individual:
        print('saving individual prediction results')
        torch.save({
                'epoch': bestepoch,
                'true_labels' : true_1,
                'pred_labels' : tk,
                'label_dict' : label_dict,
                'index' : test_df.index
                }, f"{save_path}.pth")


                                     
    # check all test examples with all metrics
    all_logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    labels = labels.detach().cpu()
    probits = F.softmax(all_logits, dim=1)
    probits = probits.detach().cpu()
    top1, top5 = utils.obs_topK(labels, probits, K=5) 
    print(f"top 1 is {top1} and top5 is {top5}")
    _, top30 = utils.obs_topK(labels, probits, K=30) 
    spectop1 = utils.species_topK(labels, probits, K=1)
    spectop5 = utils.species_topK(labels, probits, K=5)
    spectop30 = utils.species_topK(labels, probits, K=30)
    weighttop1 = utils.rarity_weighted_topK(labels, probits, K=1)
    weighttop5 = utils.rarity_weighted_topK(labels, probits, K=5)
    weighttop30 = utils.rarity_weighted_topK(labels, probits, K=30)
    # make sure at least one of the frequent species is actually present in the test set
    checkfar =  sum([sp in test_dset.far for sp in  test_df[args.to_classify].unique()]) > 1
    if checkfar:
        fartop1 = utils.subset_topK(labels, probits, test_dset.farlabs, 1)
        fartop5 = utils.subset_topK(labels, probits, test_dset.farlabs, 5)
        fartop30 = utils.subset_topK(labels, probits, test_dset.farlabs, 30)
    else:
        fartop1, fartop5, fartop30 = np.nan, np.nan, np.nan

    # make sure at least one of the common species is actually present in the test set
    checkcar =  sum([sp in test_dset.car for sp in  test_df[args.to_classify].unique()]) > 1
    if checkcar:
        cartop1 = utils.subset_topK(labels, probits, test_dset.carlabs, 1)
        cartop5 = utils.subset_topK(labels, probits, test_dset.carlabs, 5)
        cartop30 = utils.subset_topK(labels, probits, test_dset.carlabs, 30)
    else:
        cartop1, cartop5, cartop30 = np.nan, np.nan, np.nan
    # make sure at least one of the rare species is actually present in the test set
    checkrar =  sum([sp in test_dset.rar for sp in test_df[args.to_classify].unique()]) > 1
    if checkrar:
        rartop1 = utils.subset_topK(labels, probits, test_dset.rarlabs, 1)
        rartop5 = utils.subset_topK(labels, probits, test_dset.rarlabs, 5)
        rartop30 = utils.subset_topK(labels, probits, test_dset.rarlabs, 30)
    else:
        rartop1, rartop5, rartop30 = np.nan, np.nan, np.nan

    # grab ecoregion metrics
    eco_results1 = {}
    eco_results5 = {}
    for ecoregion, idxs in test_dset.l2_ecoregion.items():
        sublabels = labels[idxs]
        subpreds = probits[idxs]
        e1, e5 = utils.obs_topK(sublabels, subpreds, K=5)
        eco_results1[f"{ecoregion}_top_1"] = e1
        eco_results5[f"{ecoregion}_top_5"] = e5
    etop1 = np.mean(list(eco_results1.values()))
    etop5 = np.mean(list(eco_results5.values()))



    # grab land use category metrics
    luc_results1 = {}
    luc_results5 = {}
    for ecoregion, idxs in test_dset.land_use.items():
        sublabels = labels[idxs]
        subpreds = probits[idxs]
        e1, e5 = utils.obs_topK(sublabels, subpreds, K=5)
        luc_results1[f"luc_{ecoregion}_top_1"] = e1
        luc_results5[f"luc_{ecoregion}_top_5"] = e5
    luctop1 = np.mean(list(luc_results1.values()))
    luctop5 = np.mean(list(luc_results5.values()))
    # save out the results to CSV
    results = {
            'best_epoch' : [bestepoch],
            'model' : [hyperparams.model],
            'exp_id' : [args.exp_id],
            'test_partition' : [args.test_partition],
            'use_all_test' : [args.use_entire_split],
            'train_partition' : [hyperparams.train_partition],
            'train_partition_size' : [hyperparams.train_partition_size],
            'randomize_partition' : [hyperparams.randomize_partition],
            'train_type' : [hyperparams.train_type],
            'learning_rate' : [hyperparams.learning_rate],
            'batch_size' : [hyperparams.batch_size],
            'optimizer' : [hyperparams.optimizer],
            'date' : [datetime.now()],
            'node' : [socket.gethostname()],
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
            'jsd_patr_pbte' : [jsd_patr_pbte],
            'jsd_patr_pate' : [jsd_patr_pate]
           }
    res = pd.DataFrame({**results, **eco_results1, **eco_results5, **luc_results1, **luc_results5})
    # save accs to aggregate csv
    if not os.path.exists(total_csv):
        res.to_csv(f"{save_path}.csv", index=False)
    else:
        # WARNING: can cause race conditions if multiple
        # inference runs with the same partitions are running in parallel
        tot_csv = pd.read_csv(f"{save_path}.csv")
        res = pd.concat([tot_csv, res])
        res.to_csv(f"{save_path}.csv", index=False)




# ----------------- Runner ----------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify cli arguments.", allow_abbrev=True)

    parser.add_argument('--device', type=int, help='what device number to use (-1 for cpu)', default=-1)
    parser.add_argument("--data_dir", type=str, help="Location of directory where train/test/val split are saved to.", required=True) 
    parser.add_argument("--model_dir", type=str, help="Location of directory where model weights are saved to.", required=True) 
    parser.add_argument('--dataset', type=str, help='DivShift', default='DivShift')
    parser.add_argument("--exp_id", type=str, help="Experiment name of trained model.", required=True)
    parser.add_argument("--csv_id", type=str, help="Additional name for csv if needed.", default=None)
    parser.add_argument("--test_batch_size", type=int, help="Examples per batch", default=256)
    parser.add_argument('--processes', type=int, help='Number of workers for dataloader.', default=0)
    parser.add_argument('--train_partition', type=str, help="which split the saved weights were trained on", required=True)
    parser.add_argument('--test_partition', type=str, help="which split to test on", required=True)
    parser.add_argument('--use_entire_split', action='store_true', help='for splits with few observations, opt to use the entire split')
    parser.add_argument('--to_classify', type=str, help="which column to classify", default='name')
    parser.add_argument('--save_individual', action='store_true', help='Whether to save individual predictions and labels')
    parser.add_argument('--testing', action='store_true', help='Only run the first 50 batches')
    
    args = parser.parse_args()
    inference(args, args.test_partition, True, args.use_entire_split, args.csv_id)
