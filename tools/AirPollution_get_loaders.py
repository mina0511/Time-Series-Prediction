import numpy as np
import pandas as pd
from tqdm import trange

import torch
from torch import FloatTensor
from torch.utils.data import DataLoader



class CustomDataset:
    def __init__(self, data: torch.Tensor, window_size: int):

        self.X = []; X_append = self.X.append
        self.y = []; y_append = self.y.append
        self.length = data.shape[0] - window_size + 1

        for i in trange(self.length):
            X_append(data[i:i+window_size, 1:])
            y_append(data[i+window_size-1, 0])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_loaders(window_size=60, test_size=0.2, batch_size=16):
    data = pd.read_csv('data/AirPollution/air_pollution.csv').iloc[:, 1:].values
    data = FloatTensor(data)

    train_dataset = CustomDataset(data[:-int(data.shape[0] * test_size), :], window_size)
    test_dataset = CustomDataset(data[-int(data.shape[0] * test_size):, :], window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4*batch_size, shuffle=False)

    return train_loader, test_loader


def get_loaders_zero(window_size=60, test_size=0.2, batch_size=16):
    data = pd.read_parquet('data/AirPollution/air_pollution_zero.parquet').values
    data = FloatTensor(data)

    train_dataset = CustomDataset(data[:-int(data.shape[0] * test_size), :], window_size)
    test_dataset = CustomDataset(data[-int(data.shape[0] * test_size):, :], window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4*batch_size, shuffle=False)

    return train_loader, test_loader
