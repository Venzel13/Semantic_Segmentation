import os
os.chdir("segmentation")
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import DATA_PATH
from model import LeafModule
from preproc import LeafDataModule

log = '/home/eduard_kustov/segmentation/lightning_logs/version_2/checkpoints/epoch=31-step=383.ckpt'

model = LeafModule.load_from_checkpoint(log)
data = LeafDataModule(DATA_PATH)
trainer = pl.Trainer()
trainer.test(model, data)
#trainer.predict