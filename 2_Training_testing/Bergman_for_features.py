
# 
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pickle
with open(parent_dir+'/Dataset/Bergman_cyclization.pkl', 'rb') as file:
    data_list = pickle.load(file)

# 
from torch.utils.data import TensorDataset, DataLoader,random_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import pandas as pd
from script.model_branch import *

def get_datalist(data_list,atom_list):
    data_size = len(data_list)
    atom_num = len(atom_list)
    x3d_total = torch.empty(data_size, atom_num, 8, 8, 8)  
    x2d_total = torch.empty(data_size, atom_num, 10, 20)
    x1d_total = torch.empty(data_size, atom_num, 22)
    x0d_total = torch.empty(data_size, 13)  # 
    y_total = torch.empty(data_size, 1)  
    for idx, G in enumerate(data_list):
        x3d_total[idx] = G.x3d[atom_list, :, :, :]
        x2d_total[idx] = G.x2d[atom_list, :, :]
        x1d_total[idx] = G.x[atom_list, :22]
        x0d_total[idx] = G.global_features

        y_total[idx] = G.y

    dataset = TensorDataset(x3d_total, x2d_total, x1d_total, x0d_total, y_total)
    return dataset


def run_Cov2D(train_dataloader,test_dataloader):
    model=Cov2D().to(device)
    train_loss_list = []
    val_loss_list = []
    LR = 0.01
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)  
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)  
    num_epochs = 1000
    lr_list = []  
    for epoch in tqdm(range(num_epochs)):
        model.train()  
        running_loss = 0.0
        for x3d, x2d, x1d, x0d, targets  in train_dataloader:  
            x3d, x2d, x1d, targets = x3d.to(device), x2d.to(device), x1d.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(x2d)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_dataloader)
        scheduler.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_lr)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x3d, x2d, x1d, x0d, targets  in test_dataloader:
                x3d, x2d, x1d, targets = x3d.to(device), x2d.to(device), x1d.to(device), targets.to(device)
                outputs = model(x2d)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(test_dataloader)
        train_loss_list.append(avg_loss)
        val_loss_list.append(avg_val_loss)

    return [train_loss_list,val_loss_list]

def run_Cov1D(train_dataloader,test_dataloader):
    model=Cov1D().to(device)
    train_loss_list = []
    val_loss_list = []
    LR = 0.01
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)  
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)  
    num_epochs = 1000
    lr_list = []  
    for epoch in tqdm(range(num_epochs)):
        model.train()  
        running_loss = 0.0
        for x3d, x2d, x1d, x0d, targets  in train_dataloader: 
            x3d, x2d, x1d, targets = x3d.to(device), x2d.to(device), x1d.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(x1d)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_dataloader)
        scheduler.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_lr)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x3d, x2d, x1d, x0d, targets  in test_dataloader:
                x3d, x2d, x1d, targets = x3d.to(device), x2d.to(device), x1d.to(device), targets.to(device)
                outputs = model(x1d)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(test_dataloader)
        train_loss_list.append(avg_loss)
        val_loss_list.append(avg_val_loss)

    return [train_loss_list,val_loss_list]

def run_Cov3D(train_dataloader,test_dataloader):
    model=Cov3D().to(device)
    train_loss_list = []
    val_loss_list = []
    LR = 0.01
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3) 
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9) 
    num_epochs = 1000
    lr_list = [] 
    for epoch in tqdm(range(num_epochs)):
        model.train()  
        running_loss = 0.0
        for x3d, x2d, x1d, x0d, targets  in train_dataloader:  
            x3d, x2d, x1d, targets = x3d.to(device), x2d.to(device), x1d.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(x3d)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss /  len(train_dataloader)
        scheduler.step()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_lr)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x3d, x2d, x1d, x0d, targets  in test_dataloader:
                x3d, x2d, x1d, targets = x3d.to(device), x2d.to(device), x1d.to(device), targets.to(device)
                outputs = model(x3d)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(test_dataloader)
        train_loss_list.append(avg_loss)
        val_loss_list.append(avg_val_loss)

    return [train_loss_list,val_loss_list]


model_performance = []
for xasdfa in tqdm(range(50)):
    atom_list=[0,1,2,3,4,5,6,7,8,9]
    atom_num = len(atom_list)
    dataset = get_datalist(data_list,atom_list)
    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    criterion = nn.MSELoss()
    tmp1 = run_Cov1D(train_dataloader,test_dataloader)
    tmp2 = run_Cov2D(train_dataloader,test_dataloader)
    tmp3 = run_Cov3D(train_dataloader,test_dataloader)
    model_performance.append([tmp1,tmp2,tmp3])
    
model_performance=np.array(model_performance)
min_model = np.min(model_performance[:,:,1] ,axis=2, keepdims=True)
df = pd.DataFrame(min_model.reshape(50, 3))
df.to_csv('Bergman_feature.csv')





