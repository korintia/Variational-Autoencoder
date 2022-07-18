import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset import MNISTDataset
import torch


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
        self.val_test_size = [5000, 5000]

    def setup(self, stage=None):
        self.train_dataset = MNISTDataset("training")
        self.val_dataset, self.test_dataset = random_split(
            MNISTDataset("test"), self.val_test_size, generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
