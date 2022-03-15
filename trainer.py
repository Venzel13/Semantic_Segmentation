import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from model import LeafModule
from preproc import LeafDataModule
from config import DIR_PATH


model = LeafModule()
data = LeafDataModule(DIR_PATH)
trainer = pl.Trainer(
    gpus=1,
    callbacks = [EarlyStopping('val_loss')],
    # log_every_n_steps = 1,
)
trainer.fit(model, data)