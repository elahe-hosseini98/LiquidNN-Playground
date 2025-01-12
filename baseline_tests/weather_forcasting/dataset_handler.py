import torch
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return X.unsqueeze(-1), y

