import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.ModuleList(
            [
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 8, 5),
                nn.ReLU(),
                nn.Conv2d(8, 8, 7),
                nn.ReLU(),
                nn.Conv2d(8, 8, 7),
                nn.ReLU(),
                nn.Conv2d(8, 4, 9),
            ]
        )

    def forward(self, x):
        # print()
        # print(x.shape)
        for m in self.net:
            x = m(x)
        #     print(m)
        #     print(x.shape)
        # print()
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.ModuleList(
            [
                nn.ConvTranspose2d(4, 8, 9),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 8, 7),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 8, 7),
                nn.ReLU(),
                nn.ConvTranspose2d(8, 8, 5),
                nn.ReLU(),
                nn.Conv2d(8, 1, 3, padding=1),
                nn.Sigmoid(),
            ]
        )

    def forward(self, x):
        # print()
        # print(x.shape)
        for m in self.net:
            x = m(x)
        #     print(m)
        #     print(x.shape)
        # print()
        return x
