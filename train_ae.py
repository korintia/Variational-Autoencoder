import pytorch_lightning as pl
from data_module import MNISTDataModule
from auto_encoder import AutoEncoder
from pytorch_lightning import loggers as pl_loggers

datamodule = MNISTDataModule(batch_size=64)
model = AutoEncoder()
trainer = pl.Trainer(
    gpus=1,
    max_epochs=50,
    accumulate_grad_batches=2,
    logger=pl_loggers.TensorBoardLogger(save_dir="logs/", name=model.__class__.__name__),
)
trainer.fit(model, datamodule=datamodule)
