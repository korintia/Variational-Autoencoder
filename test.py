from sklearn import datasets
import torch
from dataset import MNISTDataset

dataset = MNISTDataset('training')
print(dataset[0].shape)
print(len(dataset))

