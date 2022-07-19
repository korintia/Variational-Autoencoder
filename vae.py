import pytorch_lightning as pl
from torch import nn
import torch
from torch import optim
import torch.nn.functional as F
from vae_network import Encoder, Decoder
from torchvision.utils import make_grid


class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        kernel_size = 4  # (4, 4) kernel
        init_channels = 8  # initial number of filters
        image_channels = 1  # MNIST images are grayscale
        latent_dim = 16  # latent dimension for sampling
        self.lr = 0.0001
        self.save_hyperparameters()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        y = batch
        y_hat = self.forward(y)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch
        y_hat = self.forward(y)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("val/loss", loss)
        if batch_idx == 0:

            self.logger.experiment.add_image("y", make_grid(y[:4]), self.current_epoch)
            self.logger.experiment.add_image("y_hat", make_grid(y_hat[:4]), self.current_epoch)

    def test_step(self, batch, batch_idx):
        y = batch
        y_hat = self.forward(y)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log("test/loss", loss)

    # def validation_epoch_end(self, outputs):
    #     val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
    #     # show val_acc in progress bar but only log val_loss
    #     results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
    #                'val_loss': val_loss_mean.item()}
    #     return results
