from sklearn import datasets
import torch
from Dataset import MNISTDataset

dataset = MNISTDataset('training')
print(dataset[0].shape)
print(len(dataset))

