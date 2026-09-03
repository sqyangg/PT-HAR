# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 21:55:10 2023

@author: sqyan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:36:55 2023

@author: sqyan
"""
#from yacs.config import CfgNode as CN

import os
import random

import numpy as np
import torch
import torch.nn as nn
import argparse
from dataset import CSI_Dataset
from dataset import *
from PTNet import ptnet50,ptnet18

import time

from thop import profile
def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    best_fit = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            
            
        
            start_time = time.time()
            outputs = model(inputs)
            
            end_time = time.time()
            #print(f"Inference time : {end_time-start_time}")
            
            
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)

            loss = criterion(outputs,labels) 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))

    return


def test(model, tensor_loader, criterion, device):
    model.eval()

    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        
        '''
        flops, params = profile(model, inputs=(inputs, ))
        
        print(flops/1000000000.0)
        print(params/1000000.0)
        '''    
            
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        start_time = time.time()
        outputs = model(inputs)
        
        end_time = time.time()
        #print(f"Inference time : {end_time-start_time}")
        
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))

    return test_acc
 


if __name__ == "__main__":
        
        root = '/Data/' 
        parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
        parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi_HAR']) 
        
        parser.add_argument('--model', choices = ['ptnet50','ptnet18']) 
        
        args = parser.parse_args()
    
        args.dataset = 'UT_HAR_data'
    
        if args.dataset == 'UT_HAR_data':
            data = UT_HAR_dataset(root)
            train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
            test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
            train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True, drop_last=True) # drop_last=True
            test_loader = torch.utils.data.DataLoader(test_set,batch_size=128,shuffle=False)
            #print(model)
            num_classes = 7
            train_epoch = 200
        else:
            print('using dataset: NTU-Fi_HAR')
            
            train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/'), batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/'), batch_size=64, shuffle=False)
            
            num_classes = 6
            train_epoch = 50

        

        if args.dataset == 'ptnet50':
            model = ptnet50(num_classes, args.dataset)
        else:
            model = ptnet18(num_classes, args.dataset)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss();
        IF_LABELSMOOTH = 'on'  ##use label smooth
        
        train(
            model=model,
            tensor_loader= train_loader,
            num_epochs= train_epoch,
            learning_rate=1e-3,
            criterion=criterion,
            device=device
             )
        
        test_acc1 = test(
            model=model,
            tensor_loader=test_loader,
            criterion=criterion,
            device= device
            )
        
       # torch.save(model.state_dict(),"NTU{:.1f}{:.4f}.pth".format(float(ii),float(test_acc1)))

        
