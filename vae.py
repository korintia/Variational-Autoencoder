import pytorch_lightning as pl
from torch import nn
import torch

# from image_plotting_callback import ImageSampler
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

from pytorch_lightning.callbacks import EarlyStopping
from torch import nn, optim, rand, sum as tsum, reshape, save
import torch.nn.functional as F
from pytorch_lightning import seed_everything, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset


class VAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()
        kernel_size = 4  # (4, 4) kernel
        init_channels = 8  # initial number of filters
        image_channels = 1  # MNIST images are grayscale
        latent_dim = 16  # latent dimension for sampling
        self.save_hyperparameters()
        self.data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

        # encoder
        self.encoder_1 = (
            nn.Conv2d(
                in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1
            ),
        )

        self.encoder_2 = (
            nn.Conv2d(
                in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1
            ),
        )

        self.encoder_3 = (
            nn.Conv2d(
                in_channels=init_channels * 2,
                out_channels=init_channels * 4,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
            ),
        )

        self.encoder_4 = nn.Conv2d(
            in_channels=init_channels * 4, out_channels=64, kernel_size=kernel_size, stride=2, padding=0
        )

        # fully connected layers for learning representations
        self.fc1 = (nn.Linear(64, 128),)
        self.fc_mu = (nn.Linear(128, latent_dim),)
        self.fc_log_var = (nn.Linear(128, latent_dim),)
        self.fc2 = nn.Linear(latent_dim, 64)

        # decoder
        self.decod_1 = (
            nn.ConvTranspose2d(
                in_channels=64, out_channels=init_channels * 8, kernel_size=kernel_size, stride=1, padding=0
            ),
        )
        self.decode_2 = (
            nn.ConvTranspose2d(
                in_channels=init_channels * 8,
                out_channels=init_channels * 4,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
            ),
        )
        self.decode_3 = (
            nn.ConvTranspose2d(
                in_channels=init_channels * 4,
                out_channels=init_channels * 2,
                kernel_size=kernel_size,
                stride=2,
                padding=1,
            ),
        )
        self.decode_4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2, out_channels=image_channels, kernel_size=kernel_size, stride=2, padding=1
        )

    def forward(self, x):
        # encoding
        x = F.relu(self.encoder_1(x))
        x = F.relu(self.encoder_2(x))
        x = F.relu(self.encoder_3(x))
        x = F.relu(self.encoder_4(x))
        batch, _, _, _ = x.shape  # ?
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)

        # decoding
        x = F.relu(self.decod_1(z))
        x = F.relu(self.decode_2(x))
        x = F.relu(self.decode_3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, log_var

    def train_dataloader(self):
        mnist_train = datasets.MNIST("./data/", download=True, train=True, transform=self.data_transform)
        return DataLoader(mnist_train, batch_size=64)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def val_dataloader(self):
        mnist_val = datasets.MNIST("./data/", download=True, train=False, transform=self.data_transform)
        return DataLoader(mnist_val, batch_size=64)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def training_step(self,batch,batch_idx):
        x,_ = batch
        batch_size = x.size(0)
        x = x.view(batch_size,-1)
        mu,log_var = self.encode(x)

        kl_loss =  (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = 1)).mean(dim =0)        
        hidden = self.reparametrize(mu,log_var)
        x_out = self.decode(hidden)
    
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x,x_out)
        #print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss
        self.log('train_loss',loss,on_step = True,on_epoch = True,prog_bar = True)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self,batch,batch_idx):
        x,_ = batch
        batch_size = x.size(0)
        x = x.view(batch_size,-1)
        mu,log_var = self.encode(x)

        kl_loss =  (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = 1)).mean(dim =0)        
        hidden = self.reparametrize(mu,log_var)
        x_out = self.decode(hidden)

        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x,x_out)
        #print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss
        self.log('val_kl_loss',kl_loss,on_step = True,on_epoch = True)
        self.log('val_recon_loss',recon_loss,on_step = True,on_epoch = True)
        self.log('val_loss',loss,on_step = True,on_epoch = True)
        return {'val_loss': loss, 'log': {'val_loss': loss}}

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o["val_loss"] for o in outputs]) / len(outputs)
        # show val_acc in progress bar but only log val_loss
        results = {
            "progress_bar": {"val_loss": val_loss_mean.item()},
            "log": {"val_loss": val_loss_mean.item()},
            "val_loss": val_loss_mean.item(),
        }
        return results


def train():
    pl.seed_everything(42)
    # leanring parameters
    epochs = args["epochs"]
    batch_size = 64
    lr = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    # train and validation data
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Initializing Dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # construct the argument parser and parser the arguments
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", default=10, type=int, help="number of epochs to train the VAE for")
    args = vars(parser.parse_args())

    model = model.LinearVAE().to(device)
    criterion = nn.BCELoss(reduction="sum")

    model = VAE().to(device)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="auto")
    # Initializing Trainer and setting parameters
    trainer = Trainer(
        gpus=1,
        auto_lr_find=False,
        max_epochs=100,
        min_epochs=1,
        auto_scale_batch_size=False,
        progress_bar_refresh_rate=10,
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model, train_loader, val_loader)
    save(model.state_dict(), "Location of our saved model")


if __name__ == "__main__":
    train()

# ===============================================
if __name__ == "__main__":

    device = "cpu"
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="auto")
    model = OurModel().to(device)
    trainer = Trainer(
        max_epochs=100,
        min_epochs=1,
        auto_lr_find=False,
        auto_scale_batch_size=False,
        progress_bar_refresh_rate=10,
        early_stop_callback=early_stop_callback,
    )
    trainer.fit(model)
    save(model.state_dict(), "/home/kt/dev/Variational-Autoencoder/models/")
