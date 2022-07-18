import pytorch_lightning as pl
from data_module import MNISTDataModule
from auto_encoder import AutoEncoder

datamodule = MNISTDataModule(batch_size=64)
trainer = pl.Trainer(
    gpus=1,
    max_epochs=50,
    accumulate_grad_batches=2,
)
model = AutoEncoder()
trainer.fit(model, datamodule=datamodule)