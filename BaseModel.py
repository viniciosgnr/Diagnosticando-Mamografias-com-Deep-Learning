# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:10:44 2019

@author: Marcos Vinicios
"""

import torch.nn as nn
import torch.nn.functional as F



class BaselineNet(nn.Module):
    
    def __init__(self,num_classes,size):
        super(BaselineNet,self).__init__()
        #above line is important
        self.layer1 = nn.Sequential()
        if(size==1024):    
            self.layer1.add_module("conv_0_1",nn.Conv2d(1,64,kernel_size=3,padding=1))
            self.layer1.add_module("relu_0",nn.ReLU())
            self.layer1.add_module("maxpool_0",nn.MaxPool2d(kernel_size=2))
            self.layer1.add_module("conv_1_1",nn.Conv2d(64,64,kernel_size=3,padding=1))
        else:
            self.layer1.add_module("conv_1_1",nn.Conv2d(1,64,kernel_size=3,padding=1))
        self.layer1.add_module("relu_1_1",nn.ReLU())
        self.layer1.add_module("conv_1_2",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer1.add_module("relu_1_1",nn.ReLU())
        self.layer1.add_module("maxpool_1",nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv_2_1",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer2.add_module("relu_2_1",nn.ReLU())
        self.layer2.add_module("conv_2_2",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer2.add_module("relu_2_1",nn.ReLU())
        self.layer2.add_module("maxpool_2",nn.MaxPool2d(kernel_size=2))
        
        
        self.layer3 = nn.Sequential()
        self.layer3.add_module("conv_3_1",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer3.add_module("maxpool_3_1",nn.MaxPool2d(kernel_size=2))
        self.layer3.add_module("conv_3_2",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer2.add_module("maxpool_3_2",nn.MaxPool2d(kernel_size=2))
        self.layer3.add_module("conv_3_3",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer3.add_module("maxpool_3_3",nn.MaxPool2d(kernel_size=2))
        self.layer3.add_module("conv_3_4",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer3.add_module("maxpool_3_4",nn.MaxPool2d(kernel_size=2))
        
        self.fc1= nn.Linear(4096,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,num_classes)
        
        self.conv_drop = nn.Dropout2d()
        self.linear_drop = nn.Dropout()
        
    def forward(self,x):
        x =  self.conv_drop(self.layer1(x))
        x =  self.conv_drop(self.layer2(x))
        x =  self.conv_drop(self.layer3(x))
        
        x = x.view(-1,self.num_flat_features(x))
        x = self.linear_drop(F.relu(self.fc1(x)))
        x = self.linear_drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size =  x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

def getModel1024(num_classes):
    return BaselineNet(num_classes,1024)

def getModel(num_classes):
    return BaselineNet(num_classes,512)

def getModel1024L(num_classes):
    return BaselineNet1024(num_classes)

def TotalParameters(model):
    return sum(p.numel() for p in model.parameters())

def TrainableTotalParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BaselineNet1024(nn.Module):
    
    def __init__(self,num_classes):
        super(BaselineNet1024,self).__init__()
        #above line is important
        self.layer1 = nn.Sequential()
        self.layer1.add_module("conv_1_1",nn.Conv2d(1,64,kernel_size=3,padding=1))
        self.layer1.add_module("relu_1_1",nn.ReLU())
        self.layer1.add_module("conv_1_2",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer1.add_module("relu_1_1",nn.ReLU())
        self.layer1.add_module("maxpool_1",nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential()
        self.layer2.add_module("conv_2_1",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer2.add_module("relu_2_1",nn.ReLU())
        self.layer2.add_module("conv_2_2",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer2.add_module("relu_2_1",nn.ReLU())
        self.layer2.add_module("maxpool_2",nn.MaxPool2d(kernel_size=2))
        
        
        self.layer3 = nn.Sequential()
        self.layer3.add_module("conv_3_1",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer3.add_module("maxpool_3_1",nn.MaxPool2d(kernel_size=2))
        self.layer3.add_module("conv_3_2",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer2.add_module("maxpool_3_2",nn.MaxPool2d(kernel_size=2))
        self.layer3.add_module("conv_3_3",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer3.add_module("maxpool_3_3",nn.MaxPool2d(kernel_size=2))
        self.layer3.add_module("conv_3_4",nn.Conv2d(64,64,kernel_size=3,padding=1))
        self.layer3.add_module("maxpool_3_4",nn.MaxPool2d(kernel_size=2))
        
        self.fc1= nn.Linear(16384,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,num_classes)
        
        self.conv_drop = nn.Dropout2d()
        self.linear_drop = nn.Dropout()
        
    def forward(self,x):
        x =  self.conv_drop(self.layer1(x))
        x =  self.conv_drop(self.layer2(x))
        x =  self.conv_drop(self.layer3(x))
        
        x = x.view(-1,self.num_flat_features(x))
        x = self.linear_drop(F.relu(self.fc1(x)))
        x = self.linear_drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self,x):
        size =  x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        # print("Flat features : {}".format(num_features))
        return num_features
    
     
     
