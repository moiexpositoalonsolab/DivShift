import pickle
import pandas as pd
import argparse
import time
import numpy as np
import torch
# BE CAREFUL with these 
# star imports!! Can accidentally
# overload things easily
from params import *
from network import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import itertools
import csv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import math
from tqdm import tqdm



# this is the method that actually does the training the model
# using the parameters stored in the parameter class, like
# learning rate, number of epochs, what model to use. Here
# you actually instantiate your dataset and your model and 
# loop through the data to train and backpropagate the model
def train_model(params):
    device = torch.device("cuda:{dev}".format(dev=ARGS.device) if ARGS.device  >= 0 else "cpu") # gross pytorch code to gpu-enable  
    batch_size=params.batch_size
    n_epochs=params.epoch    
    # here, you actually instantiate your dataset for loading in your data
    dset = Name_Of_Dataset(params.data_path)
    # here, we're also going to create a tensorboard object
    # which easily and quickly saves your loss, training accuracy etc
    # for easy visualization
    tb_writer = SummaryWriter(comment="{}".format(params.exp_id))

    # setup the model and loss
    model = network()
    # put the model on the GPU
    net.to(device)
    # set up the gradient descent optimizer
    optimizer = optim.Adam(net.parameters(), lr=params.lr)
    loss_obj = torch.nn.BCEWithLogitsLoss()
    
    # split the data into train / test
    val_split = .9       
    idxs = np.random.permutation(len(dset))
    split = int(len(idxs)*val_split)
    training_idx, test_idx = idxs[:split], idxs[split:]
    train_sampler = SubsetRandomSampler(training_idx)
    valid_sampler = SubsetRandomSampler(test_idx)
    train_dataloader = DataLoader(dset, batch_size, pin_memory=False, num_workers=params.processes, sampler=train_sampler)
    test_dataloader = DataLoader(dset, batch_size, pin_memory=False, num_workers=params.processes, sampler=test_sampler)
    overall_loss = []
    overall_acc = []
    # train loop
    
    i = 0
    while i < params.num_epochs:
        # train loop
        with tqdm(total=len(train_loader), unit="batch") as prog:
            net.train()
            for i, (data, label) in enumerate(train_loader):     
                data = data.to(device)
                output = network(data.float())
                loss = loss_obj(output, label)
                tb_writer.add_scalar("loss", loss, i)                
                overall_loss.append(loss)
                loss_obj.backward()
                optimizer.step()
    # test loop
        net.eval()
        with tqdm(total=len(test_loader), unit="batch") as prog:
            with torch.no_grad():
                for i, (data, label) in enumerate(test_loader):     
                    data = data.to(device)
                    output = network(data.float())
                    accuracy = utils.accuracy(output, label)
                    tb_writer.add_scalar("accuracy", accuracy, i)     
                    overall_acc.append(accuracy)
   
    nets_path=params.save_path
    torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, nets_path)        
    
    
# this method is what's actually run when you type 
# 'python run.py' into your terminal. Typically,
# the only thing you set up here is machine-specific
# parameters, like where on the filesystem the data is stored,
# what GPU to use, how much memory is available, etc.
# Also, this is where you want to add command-line parameters
# that you want to add for a given run. This part is optional -
# some people prefer to have parameter stored in a file and grab
# them from the file, some prefer to pass those commands from commmand line
# I prefer command line, so I will add what I have for command line parameter shere
if __name__ == "__main__":
    args = ['cli_param_1', 'cli_param_2', ...]
    # what I like to do to save CLI paramters
    # is from the params file. There, I take in
    # the arguments passed from the command-line
    # and save it in that object
    params = params(ARGS)
    train_model(params)
