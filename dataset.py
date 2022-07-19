import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self,split):
        self.data = torch.load(f'/home/kt/dev/Variational-Autoencoder/data/{split}.pt')
        self.data = self.data[0]
        self.data = self.data.float()/256

    def __getitem__(self, index):
        return self.data[index].unsqueeze(0)

    def __len__(self):
        return len(self.data)