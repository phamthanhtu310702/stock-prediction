import torch
from torch.utils.data import Dataset

class DateDataset(Dataset):
    def __init__(self, data, window, labels):   
        self.data = data
        self.window = window
        self.labels = labels
    def __getitem__(self, index):
        if (len(self.data) - index) <= self.window:
            x = self.data[index:]
        else:
            x = self.data[index: index + self.window]
        label = self.labels[index]
        return x, label

    def __len__(self):
        return len(self.data) - self.window
    
