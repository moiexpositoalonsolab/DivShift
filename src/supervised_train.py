"""
File: supervised_train.py
Authors: Elena Sierra & Lauren Gillespie
------------------
Benchmark DivShift using supervised ResNets
"""

import supervised_dataset
from supervised_test import inference
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
import glob
import socket
import random
import argparse
from tqdm import tqdm, trange
from datetime import datetime
import numpy as np
import pandas as pd
import dask.dataframe as dd
from types import SimpleNamespace
import pdb


# ----------------- Utilities ----------------- #


def save_weights(model, optimizer, epoch, freq, top1, bestacc, save_dir, steps, train_log, test_log, lr_scheduler=None):
    exp_id = save_dir.split('/')[-2]
    model_path= f"{save_dir}{exp_id}_epoch{epoch}.pth"
    # only save at X frequency
    if epoch % freq == 0:
        if lr_scheduler is None:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step' : steps,
                        'train_log': train_log,
                        'test_log': test_log
                        }, model_path)
        else:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step' : steps,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        'learning_rate' : lr_scheduler.get_last_lr(),
                        'train_log': train_log,
                        'test_log': test_log
                        }, model_path)

    if top1 > bestacc:
        if lr_scheduler is None:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step' : steps,
                        'train_log': train_log,
                        'test_log': test_log
                        }, f"{save_dir}{exp_id}_best_model.pth")
        else:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step' : steps,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        'learning_rate' : lr_scheduler.get_last_lr(),
                        'train_log': train_log,
                        'test_log': test_log
                        }, f"{save_dir}{exp_id}_best_model.pth")




# ----------------- Training ----------------- #

def train_one_epoch(args, model, device, train_loader, optimizer, epoch, logger, count, SummaryWriter):

    writer = SummaryWriter

    model.train()
    # logs loss per-batch to the console
    # stats is a range object from zero to len(train_loader)
    stats = trange(len(train_loader))
    indexer = epoch * len(train_loader)
    for batch_idx, (data, target)  in zip(stats, train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # standard softmax cross-entropy loss
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        # save train loss to tqdm bar
        stats.set_description(f'epoch {epoch}')
        stats.set_postfix(loss=loss.item())
        logger[batch_idx+indexer] = loss.item()
        writer.add_scalar('Finetune/Loss', loss.item(), count)
        count += 1
    stats.close()
    return logger



def test_one_epoch(model, device, test_loader, epoch, logger, count, SummaryWriter):
    #logging
    writer = SummaryWriter
    # need to put into eval() mode to ensure model outputs are deterministic
    model.eval()
    test_loss = 0
    correct = 0
    top5correct = 0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, total=len(test_loader),
                                 desc=f'testing epoch {epoch}'):

            data, target = data.to(device), target.to(device)
            output = model(data)
            all_logits.append(output.detach().cpu())
            all_labels.append(target.detach().cpu())

            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            writer.add_scalar('Test/Loss', (F.cross_entropy(output, target, reduction='sum').item()), count)
            count+=1
            # see if the highest predicted class was the right class
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            top5, indices = output.topk(5, dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            for c in range(indices.size(1)):
                c_tens = indices[:,c]
                top5correct += c_tens.eq(target.view_as(c_tens)).sum().item()
    # get average test loss
    test_loss /= len(test_loader.dataset)
    print(f'test set avg loss: {round(test_loss, 6)} Acc: {correct}/{len(test_loader.dataset)}:{100*round(correct / len(test_loader.dataset), 4)}%')
    logger['loss'][epoch] = round(test_loss, 6)
    logger['1accuracy'][epoch] = 100*round(correct / len(test_loader.dataset), 6)
    print(f'test set avg loss top-5: {round(test_loss, 6)} Acc: {top5correct}/{len(test_loader.dataset)}:{100*round(top5correct / len(test_loader.dataset), 4)}%')
    logger['5accuracy'][epoch] = 100*round(top5correct / len(test_loader.dataset), 6)

    all_logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    labels = labels.detach().cpu()
    probits = F.softmax(all_logits, dim=1)
    probits = probits.detach().cpu()
    top1, top5 = utils.obs_topK(labels, probits, K=5)
    spectop1 = utils.species_topK(labels, probits, K=1)
    spectop5 = utils.species_topK(labels, probits, K=5)
    logger['1spec_acc'][epoch] = spectop1
    logger['5spec_acc'][epoch] = spectop5
    return logger


def train(args, save_dir, model_weights, epoch):

    # dataset
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
        if args.randomize_partition is not None:
            # set seed for train/test split reproducibility
            rand_gen = np.random.default_rng(seed=args.randomize_partition)
            print(f"randomizing subpartitions using seed {args.randomize_partition}")
            # taxonomic bias is an edge case
            if args.train_partition in ['taxonomic_balanced', 'taxonomic_unbalanced']:
                ddf = supervised_dataset.randomize_taxonomic_train_test(ddf, rand_gen)
            # otherwise, shuffle bias partitions
            else:
                # randomize A partition
                ddf = supervised_dataset.randomize_train_test(ddf, args.train_partition, rand_gen)
                # if using it, also randomize B partition
                if args.b_partition is not None:
                    ddf = supervised_dataset.randomize_train_test(ddf, args.b_partition, rand_gen)



                    
        # get JSD between Pa train, Pb test, and Pa test
        jsd_patr_pbte, jsd_patr_pate = supervised_dataset.calculate_jsd(ddf, args.train_partition, args.test_partition, args.to_classify, args.b_partition)
        
        args.jsd_patr_pbte = jsd_patr_pbte
        args.jsd_patr_pate = jsd_patr_pate
        
        # save out JSD w/ hyperparameters
        json_fname = f'{save_dir}{args.exp_id}_hyperparams.json'
        with open(json_fname, 'w') as f:
            json.dump(vars(args), f, indent=4)

        # set up train dataset + dataloader
        train_df = ddf[ddf[args.train_partition] == 'train']
        # include B partition if relevant
        if args.b_partition is not None:
            addl_df = ddf[ddf[args.b_partition] == 'train']
            train_df = pd.concat([train_df, addl_df])


        # associate classes with index
        label_dict = {spec: i for i, spec in enumerate(sorted(train_df[args.to_classify].unique().tolist()))}
        print(f"train df is size: {train_df.shape} with {len(label_dict)} labels")


        # set up train dataloader
        train_image_dir = args.data_dir

        train_dset = supervised_dataset.LabelsDataset(train_df, train_image_dir,
                                                   label_dict, args.to_classify,
                                                   transform=transform,
                                                   target_transform=None)
        # important to drop the last batch if smaller than batch_size during forward pass
        train_loader = DataLoader(train_dset, args.batch_size,
                                  shuffle=True, num_workers=args.processes, drop_last=True)
        # Get test dataset set up
        print('setting up test dataset')
        # for training, the test dataset is dataset used to train on
        # so early stopping is done on in-distribution data
        test_df = ddf[ddf[args.train_partition] == 'test']
        # make sure to only keep observations for species seen during training
        test_df = test_df.loc[test_df[args.to_classify].isin(label_dict)]

        print(f"test df is size: {test_df.shape} with {len(test_df[args.to_classify].unique())} labels")

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

    # set up model for training
    print('setting up model')
    if args.model == 'ResNet50':
        # using imagenet pre-training
        model = models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1)
        # reset fc for our dataset size
        model.fc = nn.Linear(model.fc.in_features, len(label_dict))
    elif args.model == 'ViT-Base':
        # using imagenet pre-training
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        # reset fc for our dataset size
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(label_dict))
    elif args.model == 'ViT-Large':
        # using imagenet pre-training
        model = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1)
        # reset fc for our dataset size
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(label_dict))
    else:
        # default is ResNet18
        # using imagenet pre-training
        model = models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        # reset fc for our dataset size
        model.fc = nn.Linear(model.fc.in_features, len(label_dict))
    
    if args.train_type == 'feature_extraction':
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(label_dict))
        for param in model.fc.parameters():
            param.requires_grad = True

    params = model.parameters()
    if model_weights is not None:
        weights = torch.load(model_weights, map_location=device)
        model.load_state_dict(weights['model_state_dict'], strict=True)
    model.to(device)

    if (args.optimizer == 'Adam'):
        optimizer = optim.Adam(params, lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.learning_rate)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=args.learning_rate)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(params, lr=args.learning_rate)
    else:
        raise NotImplemented
    if model_weights is not None:
        optimizer.load_state_dict(weights['optimizer_state_dict'])
    
    print('starting training')
    # for logging purposes
    log_dir = f"{save_dir}logger"
    writer = None if args.testing else SummaryWriter(log_dir=log_dir, comment=f"{args.exp_id}")
    train_log, test_log = {}, {}
    test_log['loss'] = {}
    test_log['1accuracy'] = {}
    test_log['5accuracy'] = {}
    test_log['1spec_acc'] = {}
    test_log['5spec_acc'] = {}
    countTrainBatch = 0
    countTestBatch = 0
    best_acc = 0.0
    for epoch in range(epoch, args.num_epochs):
        print(f'starting epoch {epoch}')
        train_log = train_one_epoch(args, model, device, train_loader, optimizer, epoch, train_log, countTrainBatch, writer)
        test_log = test_one_epoch(model, device, test_loader, epoch, test_log, countTestBatch, writer)
        writer.add_scalar('Top-1 Test/Accuracy', test_log['1accuracy'][epoch], epoch)
        writer.add_scalar('Top-5 Test/Accuracy', test_log['5accuracy'][epoch], epoch)
        countTrainBatch += len(train_loader)
        countTestBatch += len(test_loader)
        save_weights(model, optimizer, epoch, args.checkpoint_freq, test_log['1spec_acc'][epoch], best_acc, save_dir, countTrainBatch, train_log, test_log)
        if test_log['1spec_acc'][epoch] > best_acc:
            best_acc = test_log['1spec_acc'][epoch]


# ----------------- Runner ----------------- #


if __name__ == "__main__":
    partition_choices = [
                    'not_city_nature',
                    'city_nature',
                    'alaska_socioeco', # 1
                    'arizona_socioeco', # 1
                    'baja_california_socioeco', # 1
                    'baja_california_sur_socioeco', # 1
                    'british_columbia_socioeco', # 1
                    'california_socioeco', # 1
                    'nevada_socioeco', # 1
                    'oregon_socioeco', # 1
                    'sonora_socioeco', # 1
                    'washington_socioeco', # 1
                    'yukon_socioeco', # 1
                    'obs_engaged',
                    'obs_casual',
                    'spatial_wilderness',
                    'spatial_modified',
                    'taxonomic_balanced',
                    'taxonomic_unbalanced',
                    'inat2021',
                    'inat2021mini',
                    'imagenet',
                    'spatial_split',
                    'random_split']

    parser = argparse.ArgumentParser(description="Specify cli arguments.", allow_abbrev=True)
    parser.add_argument("--restart", help="Use to restart model training automatically (auto) or manually (manual) by setting the exp_id to be the model of choice to finish training.", default=None, choices=['manual', 'auto'])
    parser.add_argument('--device', type=int, help='what device number to use (-1 for cpu)', default=-1)
    parser.add_argument("--data_dir", type=str, help="Location of directory where train/test/val split are saved to.", required=True)
    parser.add_argument("--model_dir", type=str, help="Location of directory where models saved to.", default='./')
    parser.add_argument('--dataset', type=str, help='DivShift', default='DivShift')
    parser.add_argument('--checkpoint_freq', type=int, help='how often to checkpoint model weights (best model is saved)', default=5)
    parser.add_argument('--optimizer', type=str, help='which optimizer (Adam, AdamW, SGD, or RMSprop)', choices=['Adam', 'AdamW', 'SGD', 'RMSprop'], default='SGD')
    parser.add_argument('--model', type=str, help='which model', choices=['ResNet18', 'ResNet50', 'ViT-Base', 'ViT-Large'], default='ResNet18')
    parser.add_argument("--exp_id", type=str, help="Experiment name for logging purposes.", required=True)
    parser.add_argument("--num_epochs", type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument("--batch_size", type=int, help="Examples per batch", default=64)
    parser.add_argument("--test_batch_size", type=int, help="Examples per batch", default=256)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer.", default=0.064)
    parser.add_argument('--processes', type=int, help='Number of workers for dataloader.', default=0)
    parser.add_argument('--randomize_partition', type=int, help="whether to use the default DivShift random partition splits (val: -1) or randomly re-assign the partitions into 80/20 train test", default=None)
    parser.add_argument('--train_type', type=str, help="Train all-layers or just final linear layer", choices=['feature_extraction', 'full_finetune'], default='full_finetune')
    parser.add_argument('--train_partition', type=str, help="which partition to train on", required=True, choices=partition_choices)
    parser.add_argument('--b_partition', type=str, help='What additional B partition to train on when using Atrain+Btrain', default=None, choices=partition_choices)
    parser.add_argument('--test_partitions', nargs='+', help="which partitions to test on")
    parser.add_argument('--save_individual', action='store_true', help='If testing the partitions, whether to save individual predictions and labels')
    parser.add_argument('--to_classify', type=str, help="which column to classify", default='name')
    parser.add_argument('--testing', action='store_true', help='dont log the run to tensorboard')
    parser.add_argument('--display_batch_loss', action='store_true', help='Display loss at each batch in the training bar')

    args = parser.parse_args()
    # restart manually by specifying full exp_id
    if args.restart == 'manual':

        save_dir = f'{args.model_dir}divshift_models/{args.exp_id}/'
        finished = glob.glob(f"{save_dir}{args.exp_id}_epoch*.pth")
        maxepoch = max([int(f.split('epoch')[-1].split('.pth')[0]) for f in finished])
        bestmodel = f"{args.model_dir}/divshift_models/{args.exp_id}/{args.exp_id}_best_model.pth" 
        bestepoch =  torch.load(bestmodel, map_location=torch.device('cpu'))['epoch']
        if bestepoch > maxepoch:
            epoch = bestepoch
            restart_epoch = 'best_model'
        else:
            epoch = maxepoch
            restart_epoch = f"epoch{epoch}"
        print(f"restarting {args.exp_id} from epoch {epoch}")
        model_weights = f"{save_dir}{args.exp_id}_{restart_epoch}.pth"
        hyperparams = f"{save_dir}{args.exp_id}_hyperparams.json"
        epoch +=1
        with open(hyperparams, 'r') as f:
            args_dict = json.load(f)
            args = SimpleNamespace(**args_dict)
    # restart automatically by starting training from most recent version
    # of this model saved to disk
    elif args.restart == 'auto':
        # get exp_id setup
        if args.b_partition is None:
            full_exp_id = f"{args.train_partition}_train_{args.exp_id}_*"
        else:
            full_exp_id = f"{args.train_partition}_{args.b_partition}_train_{args.exp_id}_*"
            
        save_dir = f'{args.model_dir}divshift_models/{full_exp_id}/'
        # pattern match any possible model trained on same setup
        possible_exps = glob.glob(f"{save_dir}")
        # get the most recent of the bunch
        exp_dir = max(possible_exps, key=os.path.getmtime)
        # get full exp_id from path
        full_exp_id = exp_dir.split('/')[-2]
        # grab what models have been trained so far
        finished = glob.glob(f"{save_dir}{full_exp_id}_epoch*.pth")
        maxepoch = max([int(f.split('epoch')[-1].split('.pth')[0]) for f in finished])
        # get what epoch the best model was at so far (and use if farther along)
        bestmodel = f"{args.model_dir}/divshift_models/{full_exp_id}/{full_exp_id}_best_model.pth" 
        bestepoch =  torch.load(bestmodel, map_location=torch.device('cpu'))['epoch']
        if bestepoch > maxepoch:
            epoch = bestepoch
            model_weights = f"{save_dir}{full_exp_id}_best_model.pth"
        else:
            epoch = maxepoch
            model_weights = f"{save_dir}{full_exp_id}_epoch{epoch}.pth"
        print(f"restarting {full_exp_id} from epoch {epoch}")
        hyperparams = f"{save_dir}{full_exp_id}_hyperparams.json"
        epoch +=1
        with open(hyperparams, 'r') as f:
            args_dict = json.load(f)
            args = SimpleNamespace(**args_dict)
            args.exp_id = full_exp_id
        
    else:
        # create dir for saving
        date = datetime.now().strftime('%Y-%m-%d')
        if args.b_partition is None:
            full_exp_id = f"{args.train_partition}_train_{args.exp_id}_{date}"
        else:
            full_exp_id = f"{args.train_partition}_{args.b_partition}_train_{args.exp_id}_{date}"
        # set up necessary variables + directories
        args.exp_id = full_exp_id
        epoch = 0
        model_weights = None
        save_dir = f'{args.model_dir}divshift_models/{full_exp_id}/'
        if not os.path.exists(save_dir):
            print(f"making dir {save_dir}")
            os.makedirs(save_dir)
    
        # save hyperparameters
        json_fname = f'{save_dir}{full_exp_id}_hyperparams.json'
        with open(json_fname, 'w') as f:
            json.dump(vars(args), f, indent=4)

    train(args, save_dir, model_weights, epoch)
    # if have partitions to test, run that inference here!
    if args.test_partitions is not None:
        # prep exp_id with correct name
        for par in args.test_partitions:
            if par not in partition_choices:
                raise ValueError(f"WARNING: {par} is not a valid partition!")
            inference(args, par)

