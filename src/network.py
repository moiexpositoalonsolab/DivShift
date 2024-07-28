import os
import numpy as np
import operator
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class network(nn.Module):
    """
    basic structure of a pytorch neural net
    All pytorch networks need an __init__ and a forward method implemented
    """
    def __init__(self, EXTRA_PARAMS_HERE):
        # this line is required for all pytorch NNs
        # it calls the nn.module superconstructor
        # to set up the backend of the network
        # TIP: make sure that the name of the class
        # in the declaration match what you give to the superconstructor
        super(network, self).__init__()
        ####### Class fields ##########
        # add class fields you need below
        # ie: self.num_channels = num_channels <- where num_channels is a parameter passed in the constructor
        ####### network layers ##########        
        # here, you add the actual layers of the network
        # the structure is as follows:
        # self.layer_name = nn.TYPEOFLAYER(inlayer, outlayer, other fields)
        # ie: self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) 
        
        
    def forward(self, INPUT_DATA):
        # this method describes how you push the data through
        # the network you described above. Specifically the param/s
        # from INPUT_DATA are pushed through each layer subsequently
        # you can consider each layer a function that acts on the input data
        # and so you call each layer initialized above with the data
        # subsequently passing it through and through, and then
        # the forward function returns the output of the network for
        # that specific bit of data
    
        # ie: 
#         x = F.relu(self.conv1(INPUT_DATA))
        return x
