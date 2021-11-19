import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from model import LeafModule
from preproc import LeafDataModule
from config import DIR_PATH, N_CLASSES


model = LeafModule(n_classes=N_CLASSES)
data = LeafDataModule(DIR_PATH)
trainer = pl.Trainer(
    callbacks = [EarlyStopping('val_loss')],
    log_every_n_steps = 1,
)
trainer.fit(model, data)