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

import pdb


# ----------------- Utilities ----------------- #


def save_weights(model, optimizer, epoch, freq, top1, bestacc, save_dir, steps, train_log, test_log, lr_scheduler=None):
    exp_id = save_dir.split('/')[-2]
    model_path= f"{save_dir}{exp_id}_epoch{epoch}.pth"
    if (epoch+1) % freq == 0:
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
    """
    IMPLEMENT UTILS FIRST
    all_logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    labels = labels.detach().cpu()
    probits = F.softmax(all_logits, dim=1)
    probits = probits.detach().cpu()
    top1, top5 = utils.obs_topK(labels, probits, K=5) 
    spectop1 = utils.species_topK(labels, probits, K=1)
    spectop5 = utils.species_topK(labels, probits, K=5)

    # hacky, but filter to common, rare, etc species
    # from auto arborist(ish)
    fartop1 = utils.subset_topK(labels, probits, traindset.farlabs, 1)
    fartop5 = utils.subset_topK(labels, probits, traindset.farlabs, 5)

    cartop1 = utils.subset_topK(labels, probits, traindset.carlabs, 1)
    cartop5 = utils.subset_topK(labels, probits, traindset.carlabs, 5)

    rartop1 = utils.subset_topK(labels, probits, traindset.rarlabs, 1)
    rartop5 = utils.subset_topK(labels, probits, traindset.rarlabs, 5)
    print(top1, top5, spectop1, spectop5, fartop1, cartop1, rartop1)

    
    if tb_writer is not None:
        tb_writer.add_scalar("test/obs_top_5", top5, epoch)
        tb_writer.add_scalar("test/obs_top_1", top1, epoch)
        tb_writer.add_scalar("test/spec_top_1", spectop1, epoch)
        tb_writer.add_scalar("test/spec_top_5", spectop5, epoch)
        tb_writer.add_scalar("test/far_top_1", fartop1, epoch)
        tb_writer.add_scalar("test/far_top_5", fartop5, epoch)
        tb_writer.add_scalar("test/car_top_1", cartop1, epoch)
        tb_writer.add_scalar("test/car_top_5", cartop5, epoch)
        tb_writer.add_scalar("test/rar_top_1", rartop1, epoch)
        tb_writer.add_scalar("test/rar_top_5", rartop5, epoch)


    # using species top 1 as early stopping
    return spectop1
    """
    return logger
    

def train(args, save_dir, full_exp_id, exp_id):

    # dataset
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
        ddf = pd.read_csv(f'{args.data_dir}/splits.csv')
        #TODO add logic for different train/test splits
        if (args.train_split in ddf.columns):
            train_df = ddf[(ddf['supervised'] == True) & 
               (ddf['download_success'] == 'yes') &
               (ddf[args.train_split] == 'train')]
        elif (args.train_split == '2019-2021'):
            train_df = ddf[(ddf['supervised'] == True) & 
               (ddf['download_success'] == 'yes') & 
               ((pd.to_datetime(ddf['date']).dt.year == 2019) | (pd.to_datetime(ddf['date']).dt.year == 2020) | (pd.to_datetime(ddf['date']).dt.year == 2021))]
        else:
            raise ValueError('Please select a valid train_split')
        
        # associate class with index
        i = 0
        for row in range(train_df.shape[0]):
            label = train_df.iloc[row][args.to_classify]
            if (label not in label_dict):
                label_dict[label] = i
                i += 1
        
        train_image_dir = args.data_dir
        
        train_dset = supervised_dataset.LabelsDataset(train_df, train_image_dir,
                                                   label_dict, args.to_classify, 
                                                   transform=transform, 
                                                   target_transform=None)
        train_loader = DataLoader(train_dset, args.batch_size, 
                                  shuffle=True, num_workers=args.processes)
        
        # Get test data
        #TODO add logic for different train/test splits
        if (args.test_split in ddf.columns):
            test_df = ddf[(ddf['supervised'] == True) & 
               (ddf['download_success'] == 'yes') &
               (ddf[args.test_split] == 'test')]
        elif (args.test_split == '2022'):
            test_df = ddf[(ddf['supervised'] == True) & (ddf['download_success'] == 'yes') & (pd.to_datetime(ddf['date']).dt.year == 2022)]
        elif (args.test_split == '2023'):
            test_df = ddf[(ddf['supervised'] == True) & (ddf['download_success'] == 'yes') & (pd.to_datetime(ddf['date']).dt.year == 2023)]
        else:
            raise ValueError('Please select a valid test_split')
        
        test_df = test_df.loc[test_df[args.to_classify].isin(label_dict)]
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
    if (args.model == 'ResNet50'):
        model = models.resnet50()
        model.load_state_dict(torch.load(args.pretrain_saved_weights)["model_state_dict"], strict=False)
    else:
        model = models.resnet18()
        model.load_state_dict(torch.load(args.pretrain_saved_weights)["model_state_dict"], strict=False)
    params = model.parameters()
    if (args.train_type == 'feature_extraction'):
        for param in params:
            param.requires_grad = False
        params = model.fc.parameters()
    model.fc = nn.Linear(model.fc.in_features, len(label_dict))
    
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

    # for logging purposes
    log_dir = f"{save_dir}logger"
    writer = None if args.testing else SummaryWriter(log_dir=log_dir, comment=f"{full_exp_id}")
    train_log, test_log = {}, {}
    test_log['loss'] = {}
    test_log['1accuracy'] = {}
    test_log['5accuracy'] = {}
    test_log['1spec_acc'] = {}
    test_log['5spec_acc'] = {}
    countTrainBatch = 0
    countTestBatch = 0
    best_acc = 0.0
    for epoch in range(args.num_epochs):
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
    parser = argparse.ArgumentParser(description="Specify cli arguments.", allow_abbrev=True)

    parser.add_argument('--device', type=int, help='what device number to use (-1 for cpu)', default=-1)
    parser.add_argument("--data_dir", type=str, help="Location of directory where train/test/val split are saved to.", required=True) 
    parser.add_argument('--dataset', type=str, help='DivShift', default='DivShift')
    parser.add_argument('--checkpoint_freq', type=int, help='how often to checkpoint model weights (best model is saved)', default=5)
    parser.add_argument('--optimizer', type=str, help='which optimizer (Adam, AdamW, SGD, or RMSprop)', choices=['Adam', 'AdamW', 'SGD', 'RMSprop'], default='SGD')
    parser.add_argument('--model', type=str, help='which model', choices=['ResNet18', 'ResNet50'], default='ResNet18')
    parser.add_argument("--exp_id", type=str, help="Experiment name for logging purposes.", required=True)
    parser.add_argument("--num_epochs", type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument("--batch_size", type=int, help="Examples per batch", default=60)
    parser.add_argument("--test_batch_size", type=int, help="Examples per batch", default=1000)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer.", default=0.001)
    parser.add_argument('--processes', type=int, help='Number of workers for dataloader.', default=0)
    parser.add_argument('--train_type', type=str, help="all-layers or one-layer", choices=['feature_extraction', 'full_finetune'], required=True)
    parser.add_argument('--train_split', type=str, help="which split to train on", required=True)
    parser.add_argument('--test_split', type=str, help="which split to test on", required=True)
    parser.add_argument('--to_classify', type=str, help="which column to classify", default='name')
    parser.add_argument('--testing', action='store_true', help='dont log the run to tensorboard')
    parser.add_argument('--display_batch_loss', action='store_true', help='Display loss at each batch in the training bar')
    
    args = parser.parse_args()
    # create dir for saving
    date = datetime.now().strftime('%Y-%m-%d')
    full_exp_id = f"{args.exp_id}_{date}"

    save_dir = f'./finetune_results/{full_exp_id}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save hyperparameters
    json_fname = f'{save_dir}{full_exp_id}_hyperparams.json'
    with open(json_fname, 'w') as f:
        json.dump(vars(args), f, indent=4)

    train(args, save_dir, full_exp_id, args.exp_id)
