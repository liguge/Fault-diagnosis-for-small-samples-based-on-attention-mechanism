# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:05:08 2021

@author: Administrator
"""
import numpy as np
import torch
import os
import re
import scipy.io as scio
import scipy.signal
from torch.utils import data as da
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
# from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
#########数据载入模块################
raw_num = 100
class Data(object):

    def __init__(self):
        self.data = self.get_data()
        self.label = self.get_label()
    def file_list(self):
        return os.listdir('./data/')
    def get_data(self):
        file_list = self.file_list()
        x = np.zeros((1024, 0))
        for i in range(len(file_list)):
            file = scio.loadmat('./data/{}'.format(file_list[i]))
            for k in file.keys():
                file_matched = re.match('X\d{3}_DE_time', k)
                #file_matched = re.match('unnamed', k)
                if file_matched:
                    key = file_matched.group()
            data1 = np.array(file[key][0:102400])     #0:80624
            for j in range(0, len(data1)-1023, 1024):
                  x = np.concatenate((x, data1[j:j+1024]), axis=1)
        return x.T
    def get_label(self):
        file_list = self.file_list()
        title = np.array([i.replace('.mat', '') for i in file_list])
        label = title[:, np.newaxis]
        label_copy = np.copy(label)
        for _ in range(raw_num-1):
            label = np.hstack((label, label_copy))
        return label.flatten()
Data = Data()
data = Data.data
label = Data.label
y = label.astype("int32")

ss = MinMaxScaler()
data = data.T
data = ss.fit_transform(data).T

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=2, stratify=y)
X_train = torch.from_numpy(X_train).unsqueeze(1)
X_test = torch.from_numpy(X_test).unsqueeze(1)
class TrainDataset(da.Dataset):
    def __init__(self):
        self.Data = X_train
        self.Label = y_train
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
class TestDataset(da.Dataset):
    def __init__(self):
        self.Data = X_test
        self.Label = y_test
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)
Train = TrainDataset()
Test = TestDataset()
train_loader = da.DataLoader(Train, batch_size=128, shuffle=True)
test_loader = da.DataLoader(Test, batch_size=10, shuffle=False)
