from re import U
import pytorch_lightning as pl
from auto_encoder import AutoEncoder
import torch
from dataset import MNISTDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
fig = plt.figure()

ae = AutoEncoder.load_from_checkpoint(
    "/home/kt/dev/Variational-Autoencoder/lightning_logs/version_19/checkpoints/epoch=49-step=23450.ckpt"
)

ae.eval()
dataset = MNISTDataset('test')
y = dataset[4]
y2 = dataset[15]
z = ae.encoder(y.unsqueeze(0))
z2 = ae.encoder(y2.unsqueeze(0))

interp_1 = z*0.8 + z2*0.2
interp_2 = z*0.5 + z2*0.5
interp_3 = z*0.2 + z2*0.8

with torch.no_grad():
    y_hat_z = ae.decoder(z)
    y_hat_1 = ae.decoder(interp_1)
    y_hat_2 = ae.decoder(interp_2)
    y_hat_3 = ae.decoder(interp_3)
    y_hat_z2 = ae.decoder(z2)

ax = fig.add_subplot(1, 5, 1)
plt.imshow(y_hat_z.squeeze(), cmap='gray')
ax = fig.add_subplot(1, 5, 2)
plt.imshow(y_hat_1.squeeze(), cmap='gray')
ax = fig.add_subplot(1, 5, 3)
plt.imshow(y_hat_2.squeeze(), cmap='gray')
ax = fig.add_subplot(1, 5, 4)
plt.imshow(y_hat_3.squeeze(), cmap='gray')
ax = fig.add_subplot(1, 5, 5)
plt.imshow(y_hat_z2.squeeze(), cmap='gray')

plt.show()